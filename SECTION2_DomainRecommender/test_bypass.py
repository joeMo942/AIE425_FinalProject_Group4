#!/usr/bin/env python3
"""Quick test for undetected-chromedriver Cloudflare bypass."""
import sys
sys.path.insert(0, '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/code')

from fetch_streamer_data import create_driver, scrape_streamer

print('Testing Cloudflare bypass with undetected-chromedriver...')
print('Creating driver...')
driver = create_driver(headless=True)

if driver is None:
    print('FAILED: Could not create driver')
    exit(1)

print('Driver created! Testing ninja page...')
result = scrape_streamer(driver, 'ninja')
driver.quit()

if result:
    print('SUCCESS! Extracted data:')
    for key, val in result.items():
        print(f'  {key}: {val}')
else:
    print('FAILED: Could not scrape ninja')
