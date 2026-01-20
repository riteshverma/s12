import asyncio
from typing import List, Dict, Any
from browserMCP.browser import BrowserSession, BrowserProfile
from browserMCP.controller.service import Controller
from browserMCP.mcp_utils.mcp_models import ActionResultOutput, ElementInfo, StructuredElementsOutput
import json
import re
from urllib.parse import urlparse
import base64
import os
from datetime import datetime
from pathlib import Path

# Global browser session (these will be accessed from the main server file)
browser_session = None
controller = None

async def ensure_browser_session():
    """Ensure browser session is initialized"""
    global browser_session, controller

    if browser_session is not None:
        try:
            if not getattr(browser_session, "initialized", False):
                await browser_session.start()
                return

            ctx = getattr(browser_session, "browser_context", None)
            ctx_closed = ctx is None or (hasattr(ctx, "is_closed") and ctx.is_closed())

            if ctx_closed:
                await stop_browser_session()
            else:
                page = getattr(browser_session, "agent_current_page", None)
                if page is None or (hasattr(page, "is_closed") and page.is_closed()):
                    await browser_session.create_new_tab()
                return
        except Exception:
            await stop_browser_session()

    if browser_session is None:
        profile = BrowserProfile(
            headless=False,
            allowed_domains=None,
            highlight_elements=True,
            bypass_csp=True,
            viewport_expansion=-1,
            include_dynamic_attributes=True,
            keep_alive=True,  # Keep browser alive between commands
        )
        browser_session = BrowserSession(profile=profile)
        controller = Controller()
        await browser_session.start()

async def execute_controller_action(action_name: str, action_params=None, **kwargs) -> ActionResultOutput:
    """Helper to execute controller actions consistently"""
    try:
        await ensure_browser_session()
        
        page = await browser_session.get_current_page()
        ActModel = controller.registry.create_action_model(page=page)
        
        # Handle actions without parameters
        if action_params is None or action_params == {}:
            action_obj = ActModel(**{action_name: {}})
        else:
            # Convert Pydantic model to dict if needed
            if hasattr(action_params, 'model_dump'):
                action_params_dict = action_params.model_dump()
            else:
                action_params_dict = action_params
            
            action_obj = ActModel(**{action_name: action_params_dict})
        
        result = await controller.act(
            action=action_obj,
            browser_session=browser_session,
            page_extraction_llm=None,
            **kwargs
        )
        result_content = result.extracted_content if hasattr(result, 'extracted_content') else None
        
        # Enhanced validation for navigation actions
        success = True
        error_msg = None
        
        if action_name in ["open_tab", "go_to_url"]:
            current_page = await browser_session.get_current_page()
            current_url = current_page.url
            
            # Check for browser error pages
            if any(error_indicator in current_url.lower() for error_indicator in [
                "chrome-error://", "about:neterror", "edge://", "about:blank"
            ]):
                success = False
                error_msg = f"Navigation failed - browser error page: {current_url}"
            
            # For open_tab, check if we have a valid domain
            elif action_name == "open_tab" and action_params:
                requested_url = action_params.url if hasattr(action_params, 'url') else str(action_params.get('url', ''))
                if requested_url and not validate_normalized_url(requested_url, current_url):
                    success = False
                    error_msg = f"Open tab failed. Requested: {requested_url}, Final: {current_url}"
        
        # Check if this is a navigation action that needs element refresh
        navigation_actions = [
            "open_tab", "go_to_url", "go_back", "search_google", 
            "click_element_by_index"  # This often causes navigation
        ]
        
        if action_name in navigation_actions and success:
            # Force refresh and get interactive elements
            state = await browser_session.get_state_summary(cache_clickable_elements_hashes=False)
            elements_result = await create_structured_elements_output(
                state.element_tree, 
                strict_mode=True
            )
            elements_json = elements_result.model_dump_json(indent=2, exclude_none=True)
            
            # Combine original result with interactive elements
            combined_content = f"{result_content}\n\nInteractive Elements:\n{elements_json}"
            
            return ActionResultOutput(
                success=success,
                content=combined_content,
                error=error_msg,
                is_done=False
            )
        
        return ActionResultOutput(
            success=success,
            content=result_content,
            error=error_msg,
            is_done=False
        )
        
    except Exception as e:
        return ActionResultOutput(success=False, error=str(e))

def categorize_element(element) -> tuple[str, str, str]:
    """
    Categorize element and determine action type
    Returns: (category, element_type, action_type)
    """
    tag_name = element.tag_name.lower()
    role = element.attributes.get('role', '').lower()
    element_type = element.attributes.get('type', '').lower()
    href = element.attributes.get('href', '')
    
    # Form elements
    if tag_name == 'input':
        if element_type in ['text', 'email', 'password', 'tel', 'url', 'search']:
            return 'form', 'text_input', 'input_text'
        elif element_type == 'number':
            return 'form', 'number_input', 'input_text'
        elif element_type == 'checkbox':
            return 'form', 'checkbox', 'click_element_by_index'
        elif element_type == 'radio':
            return 'form', 'radio_button', 'click_element_by_index'
        elif element_type == 'file':
            return 'form', 'file_upload', 'drag_drop'
        elif element_type == 'date':
            return 'form', 'date_input', 'input_text'
        elif element_type == 'submit':
            return 'form', 'submit_button', 'click_element_by_index'
    
    elif tag_name == 'textarea':
        return 'form', 'text_area', 'input_text'
    
    elif tag_name == 'select':
        return 'form', 'dropdown', 'select_dropdown_option'
    
    elif tag_name == 'button':
        if element_type == 'submit':
            return 'form', 'submit_button', 'click_element_by_index'
        elif role == 'checkbox':
            return 'form', 'toggle_button', 'click_element_by_index'
        elif role in ['tab', 'menuitem']:
            return 'interactive', 'tab_button', 'click_element_by_index'
        else:
            return 'interactive', 'button', 'click_element_by_index'
    
    # Navigation elements
    elif tag_name == 'a':
        if href and href not in ['#', 'javascript:void(0)', 'javascript:;']:
            if href.startswith('mailto:'):
                return 'interactive', 'email_link', 'click_element_by_index'
            elif href.startswith('tel:'):
                return 'interactive', 'phone_link', 'click_element_by_index'
            else:
                return 'navigation', 'link', 'click_element_by_index'
        else:
            # Links with no real destination - treat as interactive buttons
            return 'interactive', 'button_link', 'click_element_by_index'
    
    # Hoverable paragraph elements
    elif tag_name == 'p':
        return 'interactive', 'hoverable_text', 'hover item, not implemented yet'
    
    # Default for other interactive elements
    return 'interactive', 'clickable_element', 'click_element_by_index'

def create_element_description(element, category: str, element_type: str) -> str:
    """Generate a helpful description for the element"""
    text = element.get_all_text_till_next_clickable_element().strip()
    placeholder = element.attributes.get('placeholder', '')
    title = element.attributes.get('title', '')
    href = element.attributes.get('href', '')
    
    if category == 'navigation':
        if href.startswith('http'):
            return f"Navigate to external link: {text or href}"
        else:
            return f"Navigate to: {text or 'page section'}"
    
    elif category == 'form':
        if element_type == 'text_input':
            return f"Text input field: {placeholder or text or 'enter text'}"
        elif element_type == 'email_input':
            return f"Email input field: {placeholder or 'enter email address'}"
        elif element_type == 'password_input':
            return f"Password input field: {placeholder or 'enter password'}"
        elif element_type == 'submit_button':
            return f"Submit form: {text or 'submit'}"
        elif element_type == 'dropdown':
            return f"Dropdown menu: {text or placeholder or 'select option'}"
        elif element_type == 'checkbox':
            return f"Checkbox: {text or 'toggle option'}"
        elif element_type == 'file_upload':
            return f"File upload: {text or 'select file'}"
        
    elif category == 'interactive':
        if element_type == 'button':
            return f"Button: {text or 'click to activate'}"
        elif element_type == 'tab_button':
            return f"Tab: {text or 'switch tab'}"
        elif element_type == 'email_link':
            return f"Send email to: {href.replace('mailto:', '')}"
    
    return f"{element_type.replace('_', ' ').title()}: {text or 'interactive element'}"

async def filter_essential_interactive_elements(element_tree, strict_mode: bool = False) -> List:
    """Filter for only essential interactive elements that an LLM would want to interact with"""
    from browserMCP.dom.clickable_element_processor.service import ClickableElementProcessor
    
    all_elements = ClickableElementProcessor.get_clickable_elements(element_tree)
    essential_elements = []
    
    for element in all_elements:
        # ALWAYS filter by is_visible (for ads/noise) - no option needed
        if not element.is_visible:
            continue
            
        tag_name = element.tag_name.lower()
        href = element.attributes.get('href', '')
        
        # Include ALL interactive elements (p, div, span with click handlers)
        if tag_name in ['p', 'div', 'span']:
            text = element.get_all_text_till_next_clickable_element().strip()
            if text and len(text) > 2:  # Only include elements with meaningful text
                essential_elements.append(element)
                continue
        
        # STRICT MODE: Only allow essential form/navigation elements
        if strict_mode:
            # Essential form elements only
            if tag_name in ['input', 'textarea', 'select', 'button']:
                # Skip generic buttons without clear purpose
                if tag_name == 'button':
                    text = element.get_all_text_till_next_clickable_element().strip()
                    if not text or len(text) < 2:
                        continue
                essential_elements.append(element)
                continue
                
            # Essential navigation only (with real destinations, no external company logos)
            if tag_name == 'a' and href and href not in ['#', 'javascript:void(0)', 'javascript:;']:
                text = element.get_all_text_till_next_clickable_element().strip()
                # Skip if it's just a logo/image link with no text
                if not text and href.startswith('http'):
                    continue
                essential_elements.append(element)
                continue
        else:
            # NORMAL MODE: More permissive but still filtered
            
            # Skip useless links
            if tag_name == 'a' and href in ['#', 'javascript:void(0)', 'javascript:;', '']:
                text = element.get_all_text_till_next_clickable_element().strip()
                if not text or len(text) > 100:
                    continue
            
            # Skip duplicate company logo links (common pattern)
            if tag_name == 'a' and href and href.startswith('http'):
                text = element.get_all_text_till_next_clickable_element().strip()
                if not text:  # Logo links with no text
                    # Check if we already have this domain
                    domain = href.split('/')[2] if '/' in href[8:] else href[8:]
                    duplicate = any(
                        e.tag_name.lower() == 'a' and 
                        e.attributes.get('href', '').startswith('http') and
                        domain in e.attributes.get('href', '') and
                        not e.get_all_text_till_next_clickable_element().strip()
                        for e in essential_elements
                    )
                    if duplicate:
                        continue
            
            # Essential form elements
            if tag_name in ['input', 'textarea', 'select', 'button']:
                essential_elements.append(element)
                continue
                
            # Essential navigation elements (with real destinations)
            if tag_name == 'a' and href and href not in ['#', 'javascript:void(0)', 'javascript:;']:
                essential_elements.append(element)
                continue
                
            # Essential interactive roles
            role = element.attributes.get('role', '').lower()
            if role in ['button', 'link', 'menuitem', 'tab', 'checkbox', 'radio', 'combobox', 'searchbox', 'textbox']:
                essential_elements.append(element)
                continue
    
    return essential_elements

def create_smart_description(element, category: str, element_type: str) -> str:
    """Create enhanced description in format: Primary Text + Placeholder + (name/id)"""
    text = element.get_all_text_till_next_clickable_element().strip()
    placeholder = element.attributes.get('placeholder', '').strip()
    title = element.attributes.get('title', '').strip()
    name = element.attributes.get('name', '').strip()
    element_id = element.attributes.get('id', '').strip()
    href = element.attributes.get('href', '')
    
    # Special handling for dropdown elements
    if element_type == 'dropdown':
        # Always truncate at first newline for dropdowns
        if text and '\n' in text:
            text = text.split('\n')[0].strip()
        
        # Smart overlap detection
        if hasattr(element, 'children') and text:
            # Get all option texts
            option_texts = []
            for child in element.children:
                if hasattr(child, 'tag_name') and child.tag_name.lower() == 'option':
                    option_text = child.get_all_text_till_next_clickable_element().strip()
                    if option_text:
                        option_texts.append(option_text.lower())
            
            # Check overlap between description words and option words
            text_words = set(text.lower().replace(',', ' ').split())
            option_words = set(' '.join(option_texts).split())
            
            # If high overlap (>30% of description words are in options), description is just listing options
            overlap_ratio = len(text_words.intersection(option_words)) / len(text_words) if text_words else 0
            
            if overlap_ratio > 0.3:  # High overlap - description is redundant
                # Fall back to meaningful attributes
                if placeholder:
                    return f"{placeholder} ({name})" if name else placeholder
                elif name:
                    return f"Select {name.replace('_', ' ').title()} ({name})"
                else:
                    return "Select option"
            else:
                # Low overlap - description is meaningful, keep it
                return f"{text} ({name})" if name else text
    
    # For all other elements, use existing logic but truncate at newlines
    if text and '\n' in text:
        text = text.split('\n')[0].strip()
    
    # Rest of existing logic...
    primary_text = text or placeholder or title
    
    # Build description parts - avoid duplication
    description_parts = []
    
    # Add primary text
    if primary_text:
        description_parts.append(primary_text)
    
    # Add placeholder ONLY if different from primary text and not already included
    if placeholder and placeholder != primary_text and placeholder not in primary_text:
        description_parts.append(placeholder)
    
    # Add meaningful name/id in parentheses
    identifier = None
    if name and len(name) > 1 and not name.startswith(('formfield', 'form-', 'input-')):
        identifier = name
    elif element_id and len(element_id) > 1 and not element_id.startswith(('radix-', 'form-', 'input-')):
        identifier = element_id
    
    # Construct final description
    if description_parts:
        result = " ".join(description_parts)
        if identifier:
            result += f" ({identifier})"
    else:
        # Fallback handling
        if category == 'navigation':
            if href.startswith('mailto:'):
                result = href.replace('mailto:', '')
            elif href.startswith('http'):
                domain = href.split('/')[2] if '/' in href[8:] else href[8:]
                result = f"Link: {domain}"
            else:
                result = "Link"
        elif category == 'form':
            if element_type == 'dropdown':
                result = "Select option"
                if identifier:
                    result += f" ({identifier})"
            elif element_type in ['checkbox', 'radio_button']:
                result = "Toggle option"
                if identifier:
                    result += f" ({identifier})"
            elif element_type in ['text_input', 'email_input', 'password_input', 'number_input']:
                result = f"{element_type.replace('_', ' ').title()}"
                if identifier:
                    result += f" ({identifier})"
            elif element_type == 'submit_button':
                result = "Submit"
            else:
                result = f"{element_type.replace('_', ' ').title()}"
        elif category == 'interactive':
            if element_type == 'hoverable_text':
                result = "Hoverable text"
            else:
                result = f"{element_type.replace('_', ' ').title()}"
        else:
            result = "Interactive element"
    
    return result

async def create_structured_elements_output(element_tree, strict_mode: bool = False) -> StructuredElementsOutput:
    """Create ultra-compact structured categorized output for LLM consumption"""
    try:
        elements = await filter_essential_interactive_elements(element_tree, strict_mode)
        
        nav_elements = []
        form_elements = []
        button_elements = []
        
        for element in elements:
            category, element_type, action_type = categorize_element(element)
            smart_description = create_smart_description(element, category, element_type)
            
            # Create element info - only add options for dropdown elements
            if element_type == 'dropdown':
                # Extract dropdown options for select elements
                options = []
                try:
                    # Get option text from select element
                    option_elements = element.children if hasattr(element, 'children') else []
                    for option in option_elements:
                        if hasattr(option, 'tag_name') and option.tag_name.lower() == 'option':
                            option_text = option.get_all_text_till_next_clickable_element().strip()
                            if option_text and option_text not in ['-Select-', 'Select']:
                                options.append(option_text)
                except:
                    options = None
                
                element_info = ElementInfo(
                    id=element.highlight_index,
                    desc=smart_description,
                    action=action_type,
                    options=options if options else None
                )
            else:
                # For all other elements, don't include options field at all
                element_info = ElementInfo(
                    id=element.highlight_index,
                    desc=smart_description,
                    action=action_type
                )
            
            if category == 'navigation':
                nav_elements.append(element_info)
            elif category == 'form':
                form_elements.append(element_info)
            else:
                button_elements.append(element_info)
        
        return StructuredElementsOutput(
            success=True,
            nav=nav_elements,
            forms=form_elements,
            buttons=button_elements,
            total=len(elements)
        )
        
    except Exception as e:
        return StructuredElementsOutput(
            success=False,
            error=str(e),
            total=0
        )

async def get_browser_session():
    """Get the current browser session, ensuring it's initialized"""
    await ensure_browser_session()
    return browser_session

async def stop_browser_session():
    """Stop the browser session and clean up"""
    global browser_session, controller
    
    if browser_session is not None:
        await browser_session.stop()
        browser_session = None
        controller = None

def format_elements_for_llm(element_tree, format_type: str = "structured") -> str:
    """Simple formatter using browser-use's existing filtering"""
    
    if format_type == "structured":
        return format_structured_output(element_tree)
    else:
        return element_tree.clickable_elements_to_string(
            include_attributes=["id", "name", "placeholder", "type", "href"]
        )

def format_structured_output(element_tree) -> str:
    """Format in categories but use existing browser-use data"""
    from browserMCP.dom.clickable_element_processor.service import ClickableElementProcessor
    
    elements = ClickableElementProcessor.get_clickable_elements(element_tree)
    
    nav_elements = []
    form_elements = []
    interactive_elements = []
    
    for element in elements:
        if not element.is_visible:  # Use browser-use's visibility flag
            continue
            
        # Simple categorization using existing data
        tag = element.tag_name.lower()
        href = element.attributes.get('href', '')
        
        element_info = {
            "id": element.highlight_index,
            "type": tag,
            "text": element.get_all_text_till_next_clickable_element()[:50],
            "action": "click_element_by_index",
            "params": {"index": element.highlight_index}
        }
        
        if tag == 'a' and href and href not in ['#', '']:
            nav_elements.append(element_info)
        elif tag in ['input', 'textarea', 'select', 'button']:
            form_elements.append(element_info)
        else:
            interactive_elements.append(element_info)
    
    return json.dumps({
        "navigation": nav_elements,
        "forms": form_elements, 
        "interactive": interactive_elements
    }, indent=2)

def normalize_url(url: str) -> str:
    """
    Normalize URL by adding protocol if missing and validating format
    
    Examples:
    - "news.ycombinator.com" -> "https://news.ycombinator.com"
    - "google.com" -> "https://google.com"
    - "http://example.com" -> "http://example.com" (unchanged)
    - "https://test.com" -> "https://test.com" (unchanged)
    - "localhost:3000" -> "http://localhost:3000"
    - "127.0.0.1:8080" -> "http://127.0.0.1:8080"
    """
    if not url or not isinstance(url, str):
        return url
    
    url = url.strip()
    
    # If already has protocol, return as-is
    if url.startswith(('http://', 'https://', 'file://', 'ftp://')):
        return url
    
    # Special cases for localhost and IP addresses - use http
    if url.startswith(('localhost', '127.0.0.1', '0.0.0.0')) or re.match(r'^\d+\.\d+\.\d+\.\d+', url):
        return f"http://{url}"
    
    # For everything else, use https as default
    # Handle cases like "www.example.com" or "example.com"
    return f"https://{url}"

def validate_normalized_url(original_url: str, final_url: str) -> bool:
    """
    Validate that the browser actually navigated to the expected domain
    
    Args:
        original_url: The URL we tried to navigate to
        final_url: The URL the browser actually ended up at
    
    Returns:
        True if navigation was successful, False otherwise
    """
    if not original_url or not final_url:
        return False
    
    # Parse both URLs to get domains
    try:
        original_parsed = urlparse(normalize_url(original_url))
        final_parsed = urlparse(final_url)
        
        original_domain = original_parsed.netloc.lower()
        final_domain = final_parsed.netloc.lower()
        
        # Remove 'www.' prefix for comparison
        original_domain = original_domain.replace('www.', '')
        final_domain = final_domain.replace('www.', '')
        
        # Check for error pages
        error_indicators = [
            'chrome-error://', 'about:neterror', 'edge://', 
            'about:blank', 'data:text/html', 'chrome://new-tab'
        ]
        
        if any(indicator in final_url.lower() for indicator in error_indicators):
            return False
        
        # Check if domains match
        return original_domain == final_domain or original_domain in final_domain or final_domain in original_domain
        
    except Exception:
        return False

def save_base64_as_png(base64_data: str, prefix: str = "screenshot") -> str:
    """
    Convert base64 image data to PNG file and return the file path
    
    Args:
        base64_data: Base64 encoded image string
        prefix: Prefix for the filename (e.g., "screenshot", "snapshot")
    
    Returns:
        File path of the saved PNG image
    """
    try:
        # Create screenshots directory if it doesn't exist
        screenshots_dir = Path("media/screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
        filename = f"{prefix}_{timestamp}.png"
        filepath = screenshots_dir / filename
        
        # Remove data URL prefix if present (data:image/png;base64,)
        if base64_data.startswith('data:'):
            base64_data = base64_data.split(',', 1)[1]
        
        # Decode base64 and save as PNG
        image_bytes = base64.b64decode(base64_data)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        # Return relative path for portability
        return str(filepath)
        
    except Exception as e:
        return f"Error saving image: {str(e)}"

def get_image_info(filepath: str) -> dict:
    """Get basic info about the saved image"""
    try:
        from PIL import Image
        
        with Image.open(filepath) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "size_kb": round(os.path.getsize(filepath) / 1024, 2)
            }
    except ImportError:
        # PIL not available, return basic info
        size_bytes = os.path.getsize(filepath)
        return {
            "size_kb": round(size_bytes / 1024, 2)
        }
    except Exception:
        return {}

async def remove_browser_overlays(browser_session):
    """Remove browser automation overlays from the page"""
    try:
        page = await browser_session.get_current_page()
        await page.evaluate("""
            () => {
                // Remove browser-use and automation overlays
                const selectors = [
                    '[data-browser-use]',
                    '[class*="highlight"]', 
                    '[style*="outline"]',
                    '[style*="border: 2px"]',
                    '[style*="border: 3px"]',
                    'div[style*="position: absolute"][style*="z-index"]',
                    '[data-element-index]',
                    '[data-highlight]'
                ];
                
                selectors.forEach(selector => {
                    const elements = document.querySelectorAll(selector);
                    elements.forEach(el => el.remove());
                });
                
                // Remove inline styles that look like automation overlays
                const allElements = document.querySelectorAll('*');
                allElements.forEach(el => {
                    if (el.style.outline) el.style.outline = '';
                    if (el.style.border && (el.style.border.includes('2px') || el.style.border.includes('3px'))) {
                        el.style.border = '';
                    }
                    // Remove any background colors that look like highlights
                    if (el.style.backgroundColor && el.style.backgroundColor.includes('rgba(255, 0, 0')) {
                        el.style.backgroundColor = '';
                    }
                });
                
                return true;
            }
        """)
        return True
    except Exception as e:
        print(f"Warning: Could not remove overlays: {e}")
        return False

async def take_clean_screenshot(browser_session, full_page: bool = False, remove_overlays: bool = True):
    """Take a screenshot with optional overlay removal"""
    try:
        # Remove overlays if requested
        if remove_overlays:
            await remove_browser_overlays(browser_session)
            # Small delay to ensure overlays are removed
            import asyncio
            await asyncio.sleep(0.1)
        
        # Take the screenshot
        screenshot_base64 = await browser_session.take_screenshot(full_page=full_page)
        return screenshot_base64
        
    except Exception as e:
        # Fallback to regular screenshot if overlay removal fails
        return await browser_session.take_screenshot(full_page=full_page)
