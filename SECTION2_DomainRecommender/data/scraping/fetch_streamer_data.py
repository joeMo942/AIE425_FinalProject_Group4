"""
TwitchTracker Web Scraper for Streamer Metadata
================================================
Team Members:
- [Add your names and IDs here]

This script scrapes TwitchTracker to collect streamer metadata including:
- Bio, language, top games, rank, avg viewers, followers, etc.

Uses Selenium with headless Chrome.
Includes checkpointing to resume interrupted scraping.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
import logging
import re
import random
from typing import Dict, Optional, List

try:
    import undetected_chromedriver as uc
    UNDETECTED_AVAILABLE = True
except ImportError:
    UNDETECTED_AVAILABLE = False
    
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("[WARNING] Selenium not installed. Run: pip install selenium")

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = Path(__file__).parent.parent
OUTPUT_FILE = DATA_DIR / "streamer_metadata_fixed.csv"
CHECKPOINT_FILE = DATA_DIR / "scraper_checkpoint.json"
FAILED_FILE = DATA_DIR / "failed_streamers.txt"
STREAMERS_FILE = DATA_DIR / "rescrape_list.txt"  # Targeted rescrape list

# Scraping settings - increased delays for Cloudflare bypass
REQUEST_DELAY_MIN = 5.0  # Minimum seconds between requests
REQUEST_DELAY_MAX = 8.0  # Maximum seconds between requests
PAGE_TIMEOUT = 30  # Seconds to wait for page load
CHECKPOINT_INTERVAL = 1  # Save checkpoint every N streamers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)



# ============================================================================
# PART 1: DRIVER SETUP
# ============================================================================

def create_driver(headless: bool = True):
    """
    Create a Chrome WebDriver using undetected-chromedriver to bypass Cloudflare.
    
    undetected-chromedriver patches Chrome to avoid bot detection.
    """
    if not UNDETECTED_AVAILABLE:
        print("[ERROR] undetected-chromedriver not installed. Run: pip install undetected-chromedriver")
        return None
    
    options = uc.ChromeOptions()
    
    if headless:
        options.add_argument("--headless=new")
    
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    
    # Create undetected Chrome driver (specify version to match installed Chrome)
    driver = uc.Chrome(options=options, use_subprocess=True, version_main=142)
    
    driver.set_page_load_timeout(PAGE_TIMEOUT)
    return driver


def login_to_twitchtracker(driver: webdriver.Chrome, wait_for_manual: bool = True) -> bool:
    """
    Login to TwitchTracker using Twitch OAuth.
    This bypasses Cloudflare verification by having authenticated session.
    
    Args:
        driver: Selenium WebDriver
        wait_for_manual: If True, waits for user to manually complete OAuth
    
    Returns:
        True if login successful
    """
    print("[LOGIN] Navigating to TwitchTracker login...")
    
    try:
        # Go to TwitchTracker homepage
        driver.get("https://twitchtracker.com/")
        time.sleep(2)
        
        # Look for login/sign in button
        try:
            # Click "Login with Twitch" or similar button
            login_buttons = driver.find_elements("xpath", 
                "//*[contains(text(), 'Sign in') or contains(text(), 'Login') or contains(text(), 'Twitch')]"
            )
            
            for btn in login_buttons:
                if 'twitch' in btn.text.lower() or 'login' in btn.text.lower() or 'sign' in btn.text.lower():
                    btn.click()
                    print("[LOGIN] Clicked login button")
                    time.sleep(3)
                    break
        except Exception as e:
            print(f"[LOGIN] Could not find login button: {e}")
        
        # Check if we're on Twitch OAuth page
        if "twitch.tv" in driver.current_url:
            print("[LOGIN] On Twitch OAuth page...")
            
            if wait_for_manual:
                print("[LOGIN] Please complete Twitch login manually in the browser window...")
                print("[LOGIN] Waiting up to 60 seconds for login completion...")
                
                # Wait for redirect back to TwitchTracker
                for _ in range(60):
                    time.sleep(1)
                    if "twitchtracker.com" in driver.current_url:
                        print("[LOGIN] Redirected back to TwitchTracker!")
                        break
        
        # Verify login by checking for logged-in elements
        time.sleep(2)
        page_source = driver.page_source.lower()
        
        if 'logout' in page_source or 'profile' in page_source or 'dashboard' in page_source:
            print("[LOGIN] Successfully logged in!")
            return True
        else:
            print("[LOGIN] Login status unclear - proceeding anyway")
            return True  # Proceed even if unsure
            
    except Exception as e:
        print(f"[LOGIN] Error during login: {e}")
        return False



# ============================================================================
# PART 2: HELPER FUNCTIONS
# ============================================================================

def extract_number(text: str) -> Optional[int]:
    """Extract first number from text, handling commas and K/M suffixes."""
    if not text:
        return None
    
    text = text.strip().upper()
    
    # Handle K/M suffixes
    multiplier = 1
    if 'K' in text:
        multiplier = 1000
        text = text.replace('K', '')
    elif 'M' in text:
        multiplier = 1000000
        text = text.replace('M', '')
    
    # Extract number
    match = re.search(r'[\d,]+\.?\d*', text.replace(',', ''))
    if match:
        try:
            return int(float(match.group()) * multiplier)
        except:
            pass
    return None


def fetch_games_from_games_page(driver: webdriver.Chrome, username: str) -> list:
    """
    Fetch top games from the /games subpage where game names are visible as text.
    Returns list of game names.
    """
    url = f"https://twitchtracker.com/{username}/games"
    
    try:
        driver.get(url)
        time.sleep(random.uniform(5.0, 8.0))  # Same delay as main page for Cloudflare
        
        # Check for Cloudflare or error
        if "Just a moment" in driver.page_source:
             print(f"[BLOCKED] Cloudflare detected for {username}/games")
             return []
        if "Page not found" in driver.page_source:
             print(f"[404] Page not found for {username}/games")
             return []
        
        # Extract game names from the games table
        js_script = r"""
        return (function() {
            const games = [];
            
            // Target the games table specifically
            // Look for links that point to a game page AND have text content
            const gameLinks = document.querySelectorAll('table a[href*="/games/"]');
            
            if (gameLinks.length > 0) {
                gameLinks.forEach(link => {
                    // Get direct text content, ignoring hidden tooltips/images if possible
                    // CLEAN: remove newlines, tabs, and potential stat suffixes
                    let text = link.textContent.replace(/[\n\t\r]/g, ' ').trim();
                    
                    // Filter out stats and generic links
                    if (text && text.length > 1 && 
                        !['Games', 'More', 'All', 'View', 'Console'].includes(text) && 
                        !text.includes('%') && 
                        !text.match(/^\d+/) && // Starts with number (e.g. 1.2K)
                        !text.includes(' hrs') && !text.includes(' hr') &&
                        !games.includes(text)) {
                        games.push(text);
                    }
                });
            } else {
                // Fallback: any link with /games/ in href (e.g. valid game cards)
                const allLinks = document.querySelectorAll('a[href*="/games/"]');
                allLinks.forEach(link => {
                    let text = link.textContent.replace(/[\n\t\r]/g, ' ').trim();
                    if (text && text.length > 1 && 
                        !['Games', 'More', 'All'].includes(text) && 
                        !text.match(/^\d+/) && 
                        !games.includes(text)) {
                        games.push(text);
                    }
                });
            }
            
            return games.slice(0, 10);  // Get top 10 to be safe
        })();
        """
        
        games = driver.execute_script(js_script)
        if not games:
             # Improved fallback: get images alt text from table
             js_fallback = r"""
             const games = [];
             const imgs = document.querySelectorAll('table img');
             imgs.forEach(img => {
                 const t = img.getAttribute('original-title') || img.getAttribute('title') || img.alt;
                 if (t && t.length > 2 && !games.includes(t)) games.push(t);
             });
             return games.slice(0,10);
             """
             games = driver.execute_script(js_fallback)
             
        return games if games else []
    except Exception as e:
        print(f"[ERROR] Error fetching games page: {e}")
        return []
        logger.debug(f"Error fetching games for {username}: {str(e)[:30]}")
        return []



# ============================================================================
# PART 3: SCRAPING LOGIC
# ============================================================================

def scrape_streamer(driver: webdriver.Chrome, username: str) -> Optional[Dict]:
    """
    Scrape metadata for a single streamer from TwitchTracker.
    Extracts ALL fields matching datasetV2.csv format:
    - RANK, NAME, LANGUAGE, TYPE
    - MOST_STREAMED_GAME, 2ND_MOST_STREAMED_GAME
    - AVERAGE_STREAM_DURATION, FOLLOWERS_GAINED_PER_STREAM
    - AVG_VIEWERS_PER_STREAM, AVG_GAMES_PER_STREAM
    - TOTAL_TIME_STREAMED, TOTAL_FOLLOWERS, TOTAL_VIEWS
    - TOTAL_GAMES_STREAMED, ACTIVE_DAYS_PER_WEEK
    - MOST_ACTIVE_DAY, DAY_WITH_MOST_FOLLOWERS_GAINED
    """
    url = f"https://twitchtracker.com/{username}"
    
    try:
        driver.get(url)
        time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))
        
        if "Page not found" in driver.page_source or "404" in driver.title:
            print(f"[404] Page not found for {username}")
            return None
        
        if "Just a moment" in driver.page_source:
             print(f"[BLOCKED] Cloudflare detected for {username}")
             return None
             
        print(f"[DEBUG] Main page loaded for {username}. Running JS...")
        
        # Comprehensive JavaScript extraction
        js_script = r"""
        return (function() {
            const result = {
                rank: null,
                language: '',
                type: '',
                avg_stream_duration: null,
                followers_gained_per_stream: null,
                avg_viewers: null,
                avg_games_per_stream: null,
                total_time_streamed: null,
                total_followers: null,
                total_views: null,
                total_games_streamed: null,
                active_days_per_week: null,
                most_active_day: '',
                day_most_followers: '',
                games: []
            };
            
            const pageText = document.body.innerText;
            
            // Extract rank
            try {
                // Method 1: Specific rank element (often #rank-spot or similar)
                const rankEl = document.querySelector('#rank-spot, .rank-value');
                if (rankEl) {
                    const txt = rankEl.textContent.trim().replace(/,/g, '').replace('#','');
                    if (txt) result.rank = parseInt(txt);
                }
                
                // Method 2: Regex in text
                if (!result.rank) {
                    const rankMatch = pageText.match(/Ranked\s*#?\s*([\d,]+)/i);
                    if (rankMatch) result.rank = parseInt(rankMatch[1].replace(/,/g, ''));
                }
            } catch(e) {}
            
            // Extract language
            try {
                const langLink = document.querySelector('a[href*="/languages/"]');
                if (langLink) result.language = langLink.textContent.trim();
            } catch(e) {}
            
            // Extract type (personality, gaming, etc.)
            try {
                const typeMatch = pageText.match(/Type[:\s]+(\w+)/i);
                if (typeMatch) result.type = typeMatch[1];
                // Also check for common types in page
                if (pageText.includes('personality')) result.type = 'personality';
                else if (pageText.includes('Just Chatting')) result.type = 'personality';
            } catch(e) {}
            
            // Parse statistics from the stats grid/panels
            // TwitchTracker displays stats in labeled divs
            try {
                // Method 1: Find g-data divs (common pattern)
                const statDivs = document.querySelectorAll('.g-data, [class*="stat"], .panel-stat');
                statDivs.forEach(div => {
                    const label = div.querySelector('label, .label, small');
                    const value = div.querySelector('span, .value, strong');
                    if (label && value) {
                        const labelText = label.textContent.toLowerCase().trim();
                        const valueText = value.textContent.trim().replace(/[,]/g, '');
                        const numValue = parseFloat(valueText);
                        
                        if (labelText.includes('average stream')) result.avg_stream_duration = numValue;
                        if (labelText.includes('followers gained')) result.followers_gained_per_stream = numValue;
                        if (labelText.includes('avg viewers') || labelText.includes('average viewers')) result.avg_viewers = numValue;
                        if (labelText.includes('games per stream')) result.avg_games_per_stream = numValue;
                        if (labelText.includes('active days')) result.active_days_per_week = numValue;
                    }
                });
            } catch(e) {}
            
            // Method 2: Robust parsing for missing stats
            try {
                const bodyText = document.body.innerText;
                
                // Helper to find value by label (DOM traversal)
                function findStatValue(labelTrigger) {
                   const xpath = `.//*[contains(text(), '${labelTrigger}')]`;
                   const iterator = document.evaluate(xpath, document, null, XPathResult.ANY_TYPE, null);
                   let node = iterator.iterateNext();
                   while (node) {
                       // Check previous sibling
                       if (node.previousElementSibling) {
                           return node.previousElementSibling.textContent.trim();
                       }
                       // Check parent's previous sibling
                       if (node.parentElement && node.parentElement.previousElementSibling) {
                           return node.parentElement.previousElementSibling.textContent.trim();
                       }
                       node = iterator.iterateNext();
                   }
                   return null;
                }

                // Total Followers (often has rank prefix like #3 ... 19M)
                const tfVal = findStatValue('Total followers');
                if (tfVal) {
                    // Clean "#3 19,261,130" -> "19261130"
                    const cleanVal = tfVal.split('\n').pop().trim().replace(/,/g, '');
                    if (cleanVal.match(/^\d+$/)) result.total_followers = parseInt(cleanVal);
                }

                // Total Time Streamed
                const ttVal = findStatValue('Total hours streamed');
                if (ttVal) {
                    const cleanVal = ttVal.replace(/,/g, '');
                    if (cleanVal.match(/^\d+$/)) result.total_time_streamed = parseInt(cleanVal);
                }
                
                // Total Games Streamed
                const tgsVal = findStatValue('Total games streamed');
                if (tgsVal) {
                    const cleanVal = tgsVal.replace(/,/g, '');
                    if (cleanVal.match(/^\d+$/)) result.total_games_streamed = parseInt(cleanVal);
                }
                
                // Duration & Active Days (Regex on body text is reliable for these)
                if (!result.avg_stream_duration) {
                     const durMatch = bodyText.match(/Duration\s+([\d\.]+)\s*hrs/i);
                     if (durMatch) result.avg_stream_duration = parseFloat(durMatch[1]);
                }
                
                if (!result.active_days_per_week) {
                     const daysMatch = bodyText.match(/Active days per week\s+(\d+)/i);
                     if (daysMatch) result.active_days_per_week = parseInt(daysMatch[1]);
                }
                
                // Fallback for Avg Viewers (if not found by specific selector earlier)
                if (!result.avg_viewers) {
                     // Pattern 1: Number before text (7,818 Average viewers)
                     const avgMatch1 = bodyText.match(/([\d,]+)\s*average\s*viewers/i);
                     if (avgMatch1) result.avg_viewers = parseInt(avgMatch1[1].replace(/,/g, ''));
                     
                     // Pattern 2: Text before number (Avg viewers ● 7,220)
                     if (!result.avg_viewers) {
                        const avgMatch2 = bodyText.match(/Avg\.*\s*viewers\s*[●\-\:]?\s*([\d,]+)/i);
                        if (avgMatch2) result.avg_viewers = parseInt(avgMatch2[1].replace(/,/g, ''));
                     }
                }
                
                 // Days of week detection
                const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
                days.forEach(day => {
                    if (bodyText.includes('most active') && bodyText.includes(day)) {
                        result.most_active_day = day;
                    }
                });

            } catch(e) { console.error("Stats extraction error:", e); }
            
            // Extract games from data-original-title on images
            // Extract games from images or links
            try {
                // Method 1: ID-based (main page often has #channel-games)
                const gameImages = document.querySelectorAll('#channel-games img, [class*="game"] img, .img-game');
                gameImages.forEach(img => {
                    const title = img.getAttribute('data-original-title') || img.getAttribute('title') || img.alt || '';
                    if (title && title.length > 1 && !result.games.includes(title)) {
                        result.games.push(title);
                    }
                    // Check parent link title/text if image has no title
                    else if (img.parentElement && img.parentElement.tagName === 'A') {
                        let parentTitle = img.parentElement.getAttribute('title') || img.parentElement.textContent.replace(/[\n\t\r]/g, ' ').trim();
                        if (parentTitle && parentTitle.length > 1 && 
                            !parentTitle.match(/^\d+/) && // Avoid starting with number
                            !parentTitle.includes(' hrs') &&
                            !parentTitle.includes('%') &&
                            !result.games.includes(parentTitle)) {
                            result.games.push(parentTitle);
                        }
                    }
                });
                
                // Method 2: Link-based (common on subpages)
                if (result.games.length === 0) {
                    const gameLinks = document.querySelectorAll('a[href*="/games/"]');
                    gameLinks.forEach(link => {
                        const title = link.getAttribute('title') || link.textContent.replace(/[\n\t\r]/g, ' ').trim();
                        // Check child image
                        const img = link.querySelector('img');
                        const imgTitle = img ? (img.getAttribute('title') || img.getAttribute('alt')) : '';
                        
                        const finalTitle = title || imgTitle;
                        
                        if (finalTitle && finalTitle.length > 2 && 
                            !['Games', 'More', 'All', 'View'].includes(finalTitle) && 
                            !finalTitle.match(/^\d+/) &&
                            !result.games.includes(finalTitle)) {
                            result.games.push(finalTitle);
                        }
                    });
                }
            } catch(e) { console.error("Error extracting games:", e); }
            
            return result;
        })();
        """
        
        data = driver.execute_script(js_script)
        if not data:
            print(f"[WARNING] JS returned no data for {username}")
            return None
            
        # Get games list - always fetch from /games page for reliability
        games_list = data.get('games', [])
        
        # Always check /games page to get better game data
        print(f"[INFO] Navigating to games page for {username}...")
        games_from_page = fetch_games_from_games_page(driver, username)
        if games_from_page:
            # Merge: prefer games page data (which is sorted by time/rank)
            # Remove any duplicates from main page list that are in games page list
            games_list = [g for g in games_list if g not in games_from_page]
            # Prepend the sorted games page list
            games_list = games_from_page + games_list
        
        games_list = games_list[:5]  # Keep top 5
        
        # Build result matching datasetV2.csv columns (Modified: removed unused fields)
        result = {
            'RANK': data.get('rank'),
            'NAME': username,
            'LANGUAGE': data.get('language', ''),
            'MOST_STREAMED_GAME': games_list[0] if len(games_list) > 0 else '',
            '2ND_MOST_STREAMED_GAME': games_list[1] if len(games_list) > 1 else '',
            'AVERAGE_STREAM_DURATION': data.get('avg_stream_duration'),
            'AVG_VIEWERS_PER_STREAM': data.get('avg_viewers'),
            'TOTAL_TIME_STREAMED': data.get('total_time_streamed'),
            'TOTAL_FOLLOWERS': data.get('total_followers'),
            'TOTAL_GAMES_STREAMED': data.get('total_games_streamed'),
            'ACTIVE_DAYS_PER_WEEK': data.get('active_days_per_week'),
        }
        
        return result
        
    except TimeoutException:
        logger.debug(f"Timeout for {username}")
        return None
    except WebDriverException as e:
        logger.debug(f"WebDriver error for {username}: {str(e)[:50]}")
        return None
    except Exception as e:
        print(f"[ERROR] Error scraping {username}: {str(e)[:100]}")
        # logger.debug(f"Error scraping {username}: {e}")
        return None



# ============================================================================
# PART 4: CHECKPOINTING
# ============================================================================

def load_checkpoint() -> Dict:
    """Load checkpoint from previous run."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'results': []}


def save_checkpoint(checkpoint: Dict):
    """Save checkpoint to disk."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)


def save_failed(failed_list: List[str]):
    """Save list of failed streamers."""
    with open(FAILED_FILE, 'w') as f:
        for streamer in failed_list:
            f.write(f"{streamer}\n")


def save_results_csv(results: List[Dict]):
    """Save current results to CSV."""
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)



# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main scraping pipeline."""
    print("\n" + "=" * 60)
    print("TWITCHTRACKER WEB SCRAPER")
    print("=" * 60)
    
    if not SELENIUM_AVAILABLE:
        print("[ERROR] Selenium is required. Install with: pip install selenium")
        return
    
    # Load streamer list
    if not STREAMERS_FILE.exists():
        print(f"[ERROR] Streamer list not found: {STREAMERS_FILE}")
        print("        Run data_preprocessing.py first!")
        return
    
    with open(STREAMERS_FILE, 'r') as f:
        all_streamers = [line.strip() for line in f if line.strip()]
    
    print(f"[LOADED] {len(all_streamers):,} streamers to scrape")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed = set(checkpoint.get('completed', []))
    results = checkpoint.get('results', [])
    
    # Filter out already completed
    remaining = [s for s in all_streamers if s not in completed]
    print(f"[RESUME] {len(completed):,} already completed, {len(remaining):,} remaining")
    
    if not remaining:
        print("[DONE] All streamers already scraped!")
        if results:
            save_results_csv(results)
            print(f"[SAVED] {OUTPUT_FILE.name}")
        return
    
    # Estimate time
    avg_delay = (REQUEST_DELAY_MIN + REQUEST_DELAY_MAX) / 2
    est_seconds = len(remaining) * avg_delay
    est_hours = est_seconds / 3600
    print(f"[ESTIMATE] ~{est_hours:.1f} hours remaining")
    
    # Start scraping
    print("\n" + "-" * 40)
    print("Starting scraper... Press Ctrl+C to stop safely")
    print("-" * 40 + "\n")
    
    failed = []
    driver = None
    success_count = 0
    
    try:
        print("[INIT] Creating Chrome WebDriver (GUI mode for debugging)...")
        driver = create_driver(headless=False)  # GUI enabled per user request
        
        # SKIP LOGIN per user request
        # print("[INIT] Logging into TwitchTracker...")
        # login_to_twitchtracker(driver)
        
        print("[INIT] Starting FULL scraping...")
        print("[INIT] WebDriver ready!\n")
        
        # FULL RUN (No limit)
        # test_limit = 10
        # remaining = remaining[:test_limit]
        
        for i, username in enumerate(remaining):
            # Scrape
            data = scrape_streamer(driver, username)
            
            if data:
                results.append(data)
                completed.add(username)
                success_count += 1
                # Show progress every 10
                if success_count % 10 == 0:
                    logger.info(f"[{len(completed):,}/{len(all_streamers):,}] ✓ {username} (success: {success_count})")
            else:
                failed.append(username)
                completed.add(username)  # Mark as attempted
                if len(failed) % 50 == 1:
                    logger.warning(f"[{len(completed):,}/{len(all_streamers):,}] ✗ {username} (failed: {len(failed)})")
            
            # Checkpoint
            if len(completed) % CHECKPOINT_INTERVAL == 0:
                checkpoint = {'completed': list(completed), 'results': results}
                save_checkpoint(checkpoint)
                save_failed(failed)
                save_results_csv(results)
                logger.info(f"[CHECKPOINT] {len(results):,} results, {len(failed)} failed")
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Saving progress...")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        # Final save
        checkpoint = {'completed': list(completed), 'results': results}
        save_checkpoint(checkpoint)
        save_failed(failed)
        save_results_csv(results)
        
        if driver:
            driver.quit()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SCRAPING SUMMARY")
    print("=" * 60)
    print(f"       Total attempted: {len(completed):,}")
    print(f"       Successful:      {len(results):,}")
    print(f"       Failed:          {len(failed):,}")
    
    if results:
        print(f"\n[SAVED] {OUTPUT_FILE.name}")
        df = pd.DataFrame(results)
        print("\n       Sample data:")
        print(df[['NAME', 'LANGUAGE', 'RANK', 'AVG_VIEWERS_PER_STREAM', 'MOST_STREAMED_GAME']].head(5).to_string())


if __name__ == "__main__":
    main()
