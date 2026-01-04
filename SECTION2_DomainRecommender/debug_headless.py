#!/usr/bin/env python3
"""Debug script to inspect page content in headless mode."""
import sys
sys.path.insert(0, '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/code')

from fetch_streamer_data import create_driver
import time

print('Debugging headless mode content...')
driver = create_driver(headless=True)
print('Driver created.')

try:
    print('Navigating to ninja page...')
    driver.get("https://twitchtracker.com/ninja")
    time.sleep(10) # ample time
    
    print('Page title:', driver.title)
    
    # Check for game elements
    games = driver.execute_script("""
        const games = [];
        const images = document.querySelectorAll('#channel-games img, [class*="game"] img');
        images.forEach(img => {
            games.push(img.getAttribute('data-original-title') || img.getAttribute('title') || img.alt);
        });
        return games;
    """)
    print('Games found via JS:', games)
    
    # Check for "Sign in" messages
    src = driver.page_source
    if "Sign In" in src or "Login" in src:
        print('Login keywords found in source')
        
    # Check specific game container
    import re
    if "VALORANT" in src:
        print('VALORANT string found in source!')
    else:
        print('VALORANT string NOT found in source')

finally:
    driver.quit()
