#!/usr/bin/env python3
"""
Add a synchronous method wrapper for getting options chain
"""

import re
from pathlib import Path


def add_sync_method():
    """Add sync wrapper method"""
    return '''
    def get_options_chain_sync(self, symbol: str, min_strike: float = None, max_strike: float = None,
                              min_expiry_days: int = 7, max_expiry_days: int = 45) -> List[Dict]:
        """Synchronous wrapper for get_options_chain"""
        import asyncio
        
        async def _get():
            return await self.get_options_chain(
                symbol=symbol,
                min_strike=min_strike,
                max_strike=max_strike,
                min_expiry_days=min_expiry_days,
                max_expiry_days=max_expiry_days
            )
        
        # Run in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _get())
                    return future.result()
            else:
                return loop.run_until_complete(_get())
        except Exception as e:
            logger.error(f"Error getting options chain: {e}")
            return []
'''


def main():
    """Add sync method to AlpacaOptionsDataCollector"""
    
    collector_path = Path('src/options_data_collector.py')
    if not collector_path.exists():
        print("❌ options_data_collector.py not found!")
        return
    
    with open(collector_path, 'r') as f:
        content = f.read()
    
    # Find the class definition
    class_match = re.search(r'class AlpacaOptionsDataCollector.*?(?=\nclass|\Z)', content, re.DOTALL)
    if class_match:
        # Find the last method in the class
        last_method = re.findall(r'(\n    def \w+.*?)(?=\n    def|\nclass|\Z)', class_match.group(), re.DOTALL)
        if last_method:
            # Insert after the last method
            insert_pos = content.find(last_method[-1]) + len(last_method[-1])
            content = content[:insert_pos] + add_sync_method() + content[insert_pos:]
            
            # Write back
            with open(collector_path, 'w') as f:
                f.write(content)
            
            print("✅ Added sync method to AlpacaOptionsDataCollector")
    else:
        print("❌ Could not find AlpacaOptionsDataCollector class")


if __name__ == "__main__":
    main()