#!/usr/bin/env python3
"""Test the updated scraper."""
import sys
sys.path.insert(0, '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/code')

from fetch_streamer_data import create_driver, scrape_streamer

print('Testing updated scraper on ninja...')
driver = create_driver()
print('Driver created!')
result = scrape_streamer(driver, 'ninja')
driver.quit()

if result:
    print('SUCCESS! Extracted 17 columns:')
    for key, val in result.items():
        print(f'  {key}: {val}')
else:
    print('FAILED to scrape')
