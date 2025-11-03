"""
Quick Start Script - Launch Both Recommendation Servers

Launches:
1. Classic Server (ALS + Ridge) on port 8001
2. Neural Server (NCF + SBERT) on port 8002

Usage:
    python start_servers.py
    python start_servers.py --classic-only
    python start_servers.py --neural-only
"""
import argparse
import subprocess
import sys
from pathlib import Path
import time

def check_models_exist():
    """Check if trained models exist"""
    classic_exists = Path("./artifacts").exists()
    neural_exists = Path("./artifacts_neural").exists()
    
    return classic_exists, neural_exists

def start_classic_server():
    """Start classic recommender server"""
    print("🚀 Starting Classic Server (ALS + Ridge) on port 8001...")
    return subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def start_neural_server():
    """Start neural recommender server"""
    print("🚀 Starting Neural Server (NCF + SBERT) on port 8002...")
    return subprocess.Popen(
        [sys.executable, "server_neural.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def main():
    parser = argparse.ArgumentParser(description='Start Recommendation Servers')
    parser.add_argument('--classic-only', action='store_true',
                       help='Start only classic server')
    parser.add_argument('--neural-only', action='store_true',
                       help='Start only neural server')
    args = parser.parse_args()
    
    # Check models
    classic_exists, neural_exists = check_models_exist()
    
    print("="*70)
    print("📚 BOOK RECOMMENDATION SYSTEM - SERVER LAUNCHER")
    print("="*70)
    print(f"\n📊 Model Status:")
    print(f"  Classic (ALS + Ridge): {'✅ Found' if classic_exists else '❌ Not Found'}")
    print(f"  Neural (NCF + SBERT):  {'✅ Found' if neural_exists else '❌ Not Found'}")
    
    if not classic_exists:
        print(f"\n⚠️  Classic model not found. Train with:")
        print(f"     python train.py --evaluate")
    
    if not neural_exists:
        print(f"\n⚠️  Neural model not found. Train with:")
        print(f"     python train_neural.py --evaluate")
    
    # Start servers
    processes = []
    
    if args.neural_only:
        if not neural_exists:
            print("\n❌ Cannot start neural server - model not trained!")
            sys.exit(1)
        processes.append(('Neural', start_neural_server()))
    elif args.classic_only:
        if not classic_exists:
            print("\n❌ Cannot start classic server - model not trained!")
            sys.exit(1)
        processes.append(('Classic', start_classic_server()))
    else:
        # Start both
        if classic_exists:
            processes.append(('Classic', start_classic_server()))
        if neural_exists:
            time.sleep(2)  # Stagger startup
            processes.append(('Neural', start_neural_server()))
        
        if not processes:
            print("\n❌ No models found! Train at least one model first.")
            sys.exit(1)
    
    print("\n" + "="*70)
    print("✅ SERVERS STARTED!")
    print("="*70)
    
    if any(name == 'Classic' for name, _ in processes):
        print("\n📡 Classic Server:")
        print("   URL: http://localhost:8001")
        print("   Docs: http://localhost:8001/docs")
        print("   Model: ALS + Ridge")
        print("   Features: Online Learning ✅")
    
    if any(name == 'Neural' for name, _ in processes):
        print("\n🧠 Neural Server:")
        print("   URL: http://localhost:8002")
        print("   Docs: http://localhost:8002/docs")
        print("   Model: NCF + SBERT")
        print("   Features: Semantic Understanding ✅")
    
    print("\n💡 Tips:")
    print("   - Test Classic: python test_online_api.py")
    print("   - Test Neural:  python test_neural_api.py")
    print("   - Press Ctrl+C to stop all servers")
    
    print("\n⏳ Servers running... (Press Ctrl+C to stop)")
    
    try:
        # Wait for all processes
        for name, proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping servers...")
        for name, proc in processes:
            proc.terminate()
            print(f"   Stopped {name} server")
        print("\n✅ All servers stopped")

if __name__ == "__main__":
    main()
