#!/usr/bin/env python3
"""
Script to download Qlib data for US market
Based on the research paper's approach
"""

import os
import sys
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_qlib_data(target_dir: str = None, region: str = "us"):
    """
    Download Qlib data for specified region
    
    Args:
        target_dir: Target directory for data (default: ~/.qlib/qlib_data/<region>_data)
        region: Market region (us, cn, etc.)
    """
    if target_dir is None:
        target_dir = os.path.expanduser(f"~/.qlib/qlib_data/{region}_data")
    
    logger.info(f"Downloading Qlib data for {region} market to {target_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    
    # Download using Qlib's data script
    try:
        # Install qlib if not installed
        try:
            import qlib
            logger.info("Qlib is already installed")
        except ImportError:
            logger.info("Installing Qlib...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyqlib"])
        
        # Download data
        logger.info(f"Downloading {region} market data...")
        cmd = [
            sys.executable, "-m", "qlib.run.get_data",
            "qlib_data",
            "--target_dir", target_dir,
            "--region", region
        ]
        
        if region == "us":
            # For US market, we need specific date range
            cmd.extend(["--interval", "1d"])
        
        subprocess.check_call(cmd)
        
        logger.info(f"✅ Qlib data downloaded successfully to {target_dir}")
        
        # Verify installation
        test_qlib_installation(target_dir, region)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download Qlib data: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


def test_qlib_installation(data_dir: str, region: str):
    """Test if Qlib data was installed correctly"""
    try:
        import qlib
        from qlib.data import D
        
        # Initialize Qlib
        qlib.init(provider_uri=data_dir, region=region)
        
        # Test loading some data
        test_symbols = {
            "us": ["AAPL", "MSFT", "GOOGL"],
            "cn": ["000001", "000002", "000004"]
        }
        
        symbols = test_symbols.get(region, ["AAPL"])
        
        logger.info(f"Testing data access for symbols: {symbols}")
        
        # Try to load basic data
        df = D.features(
            instruments=symbols[:1],
            fields=["$close", "$volume"],
            start_time="2022-01-01",
            end_time="2022-01-31"
        )
        
        if not df.empty:
            logger.info(f"✅ Qlib data verified successfully. Loaded {len(df)} data points.")
        else:
            logger.warning("⚠️  No data loaded. Please check if the symbols and date range are correct.")
            
    except Exception as e:
        logger.error(f"Failed to verify Qlib installation: {e}")


def main():
    parser = argparse.ArgumentParser(description='Download Qlib data for trading')
    parser.add_argument('--target-dir', type=str, default=None,
                        help='Target directory for Qlib data')
    parser.add_argument('--region', type=str, default='us',
                        choices=['us', 'cn'],
                        help='Market region to download (default: us)')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test existing installation')
    
    args = parser.parse_args()
    
    if args.test_only:
        if args.target_dir is None:
            args.target_dir = os.path.expanduser(f"~/.qlib/qlib_data/{args.region}_data")
        test_qlib_installation(args.target_dir, args.region)
    else:
        download_qlib_data(args.target_dir, args.region)
        
        print("\n" + "="*60)
        print("Qlib data download complete!")
        print("="*60)
        print("\nTo use Qlib features in training, run:")
        print(f"python train_ppo_lstm.py --use-qlib --qlib-region {args.region}")
        if args.target_dir:
            print(f"  --qlib-data-path {args.target_dir}")
        print("\nExample:")
        print("python train_ppo_lstm.py --use-qlib --episodes 1000")


if __name__ == "__main__":
    main()