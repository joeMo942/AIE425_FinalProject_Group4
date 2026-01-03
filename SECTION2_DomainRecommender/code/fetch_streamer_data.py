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
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "streamer_metadata.csv"
CHECKPOINT_FILE = DATA_DIR / "scraper_checkpoint.json"
FAILED_FILE = DATA_DIR / "failed_streamers.txt"
STREAMERS_FILE = DATA_DIR / "top_2000_streamers.txt"  # Use top 2000 for faster scraping

# Scraping settings
REQUEST_DELAY_MIN = 2.0  # Minimum seconds between requests
REQUEST_DELAY_MAX = 4.0  # Maximum seconds between requests
PAGE_TIMEOUT = 20  # Seconds to wait for page load
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N streamers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_driver() -> webdriver.Chrome:
    """Create a headless Chrome WebDriver with anti-detection measures."""
    options = Options()
    
    # Anti-detection measures
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")
    
    # Realistic user agent
    options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=options)
    
    # Additional anti-detection
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    driver.set_page_load_timeout(PAGE_TIMEOUT)
    return driver


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
        time.sleep(random.uniform(1.5, 2.5))  # Shorter delay for subpage
        
        # Check for Cloudflare or error
        if "Just a moment" in driver.page_source or "Page not found" in driver.page_source:
            return []
        
        # Extract game names from the games table
        js_script = """
        return (function() {
            const games = [];
            
            // Method 1: Look for table rows with game names
            const rows = document.querySelectorAll('table tr, .game-row, [class*="game"]');
            rows.forEach(row => {
                const links = row.querySelectorAll('a');
                links.forEach(link => {
                    const text = link.textContent.trim();
                    if (text && text.length > 2 && !text.includes('Game') && 
                        !text.includes('Hours') && !text.includes('%') && 
                        !games.includes(text)) {
                        games.push(text);
                    }
                });
            });
            
            // Method 2: Find all links containing /games/ in href
            if (games.length === 0) {
                const allLinks = document.querySelectorAll('a[href*="/games/"]');
                allLinks.forEach(link => {
                    const text = link.textContent.trim();
                    if (text && text.length > 2 && !['Games', 'More', 'All'].includes(text) && 
                        !games.includes(text)) {
                        games.push(text);
                    }
                });
            }
            
            return games.slice(0, 5);  // Top 5 games
        })();
        """
        
        games = driver.execute_script(js_script)
        return games if games else []
        
    except Exception as e:
        logger.debug(f"Error fetching games for {username}: {str(e)[:30]}")
        return []


def scrape_streamer(driver: webdriver.Chrome, username: str) -> Optional[Dict]:
    """
    Scrape metadata for a single streamer from TwitchTracker.
    Uses JavaScript execution for reliable DOM extraction.
    
    Returns dict with streamer info or None if failed.
    """
    url = f"https://twitchtracker.com/{username}"
    
    try:
        driver.get(url)
        
        # Wait for page to load
        time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))
        
        # Check for 404 or error page
        if "Page not found" in driver.page_source or "404" in driver.title:
            return None
        
        # Use JavaScript to extract data from the DOM
        js_script = """
        return (function() {
            const result = {
                bio: '',
                language: '',
                rank: null,
                avg_viewers: null,
                followers: null,
                hours_streamed: null,
                peak_viewers: null,
                games: []
            };
            
            // Extract bio - text before Twitch link
            try {
                const twitchLink = document.querySelector('a[href*="twitch.tv"]');
                if (twitchLink && twitchLink.parentElement) {
                    const bio = twitchLink.parentElement.previousElementSibling;
                    if (bio && bio.textContent) {
                        result.bio = bio.textContent.trim();
                    }
                }
            } catch(e) {}
            
            // Extract language
            try {
                const langLink = document.querySelector('a[href*="/languages/"]');
                if (langLink) {
                    result.language = langLink.textContent.trim();
                }
            } catch(e) {}
            
            // Extract rank - using text content search
            try {
                const rankMatch = document.body.innerText.match(/Ranked\\s*#?\\s*(\\d+)/i);
                if (rankMatch) {
                    result.rank = parseInt(rankMatch[1]);
                }
            } catch(e) {}
            
            // Extract stats by finding labels and their sibling values
            try {
                const allDivs = document.querySelectorAll('div');
                allDivs.forEach(div => {
                    const text = div.textContent.trim();
                    const next = div.nextElementSibling;
                    if (next) {
                        const nextText = next.textContent.trim().replace(/[,●]/g, '');
                        const num = parseInt(nextText);
                        
                        if (text === 'Followers' && !isNaN(num)) {
                            result.followers = num;
                        }
                        if (text === 'Avg viewers' && !isNaN(num)) {
                            result.avg_viewers = num;
                        }
                    }
                });
            } catch(e) {}
            
            // Extract hours streamed from page text
            try {
                const hoursMatch = document.body.innerText.match(/(\\d[\\d,]*)\\s*(?:total\\s*)?hours?\\s*streamed/i);
                if (hoursMatch) {
                    result.hours_streamed = parseInt(hoursMatch[1].replace(/,/g, ''));
                }
            } catch(e) {}
            
            // Extract peak viewers
            try {
                const peakMatch = document.body.innerText.match(/peak\\s*viewers?[\\s:]*([\\d,]+)/i);
                if (peakMatch) {
                    result.peak_viewers = parseInt(peakMatch[1].replace(/,/g, ''));
                }
            } catch(e) {}
            
            // Extract game names from the games section or page text
            try {
                // Method 1: Get game names from data-original-title attribute on images
                const gameImages = document.querySelectorAll('#channel-games img, .game-image img, [class*="game"] img');
                gameImages.forEach(img => {
                    const title = img.getAttribute('data-original-title') || img.getAttribute('title') || img.getAttribute('alt');
                    if (title && title.length > 1 && !result.games.includes(title)) {
                        result.games.push(title);
                    }
                });
                
                // Method 2: Look for data-original-title on any element in games section
                const gamesSection = document.querySelectorAll('#channel-games [data-original-title], [class*="game"] [data-original-title]');
                gamesSection.forEach(elem => {
                    const title = elem.getAttribute('data-original-title');
                    if (title && title.length > 1 && !result.games.includes(title)) {
                        result.games.push(title);
                    }
                });
                
                // Method 3: Get game links with title attribute
                const gameLinks = document.querySelectorAll('#channel-games a.entity, a[href*="/games/"]');
                gameLinks.forEach(link => {
                    const title = link.getAttribute('title') || link.getAttribute('data-original-title');
                    // Also check child images
                    const img = link.querySelector('img');
                    const imgTitle = img ? (img.getAttribute('data-original-title') || img.getAttribute('title') || img.getAttribute('alt')) : null;
                    const gameName = title || imgTitle;
                    if (gameName && gameName.length > 1 && !['Games', 'More', 'All'].includes(gameName)) {
                        if (!result.games.includes(gameName)) {
                            result.games.push(gameName);
                        }
                    }
                });
            } catch(e) {}
            
            return result;
        })();
        """
        
        data = driver.execute_script(js_script)
        
        # Build initial result from main page
        games_list = data.get('games', [])
        
        # If no games found on main page, try the /games subpage
        if not games_list:
            games_list = fetch_games_from_games_page(driver, username)
        
        result = {
            'streamer_username': username,
            'bio': data.get('bio', '')[:500] if data.get('bio') else '',
            'language': data.get('language', ''),
            'rank': data.get('rank'),
            'avg_viewers': data.get('avg_viewers'),
            'followers': data.get('followers'),
            'hours_streamed': data.get('hours_streamed'),
            'peak_viewers': data.get('peak_viewers'),
            '1st_game': games_list[0] if len(games_list) > 0 else '',
            '2nd_game': games_list[1] if len(games_list) > 1 else '',
        }
        
        return result
        
    except TimeoutException:
        logger.debug(f"Timeout for {username}")
        return None
    except WebDriverException as e:
        logger.debug(f"WebDriver error for {username}: {str(e)[:50]}")
        return None
    except Exception as e:
        logger.debug(f"Error scraping {username}: {str(e)[:50]}")
        return None


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
        print("[INIT] Creating Chrome WebDriver...")
        driver = create_driver()
        print("[INIT] WebDriver ready!\n")
        
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
        print(df[['streamer_username', 'language', 'rank', 'avg_viewers', '1st_game']].head(5).to_string())


if __name__ == "__main__":
    main()
