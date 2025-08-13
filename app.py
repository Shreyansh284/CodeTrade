#!/usr/bin/env python3
"""
Optimized Stock Pattern Detector - Web Application Entry Point

A streamlined, high-performance application for stock pattern detection
with automatic browser launch and optimized settings.
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading
import signal
import atexit
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def find_free_port(start_port=8501, max_attempts=10):
    """Find a free port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return start_port  # Fallback to default

def open_browser(url, delay=3):
    """Open browser after a delay to ensure server is ready."""
    def delayed_open():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            print(f"✅ Opening browser at: {url}")
        except Exception as e:
            print(f"⚠️  Could not auto-open browser: {e}")
            print(f"📂 Please manually open: {url}")
    
    thread = threading.Thread(target=delayed_open, daemon=True)
    thread.start()

def cleanup_handler(signum=None, frame=None):
    """Clean up resources on exit."""
    print("\n🔄 Shutting down gracefully...")
    sys.exit(0)

def main():
    """Launch the optimized stock pattern detector web application."""
    print("🚀 Starting Stock Pattern Detector...")
    
    # Register cleanup handlers
    atexit.register(cleanup_handler)
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    # Find available port
    port = find_free_port()
    url = f"http://localhost:{port}"
    
    # Path to streamlit app
    streamlit_app = current_dir / "ui" / "streamlit_app.py"
    
    if not streamlit_app.exists():
        print(f"❌ Error: Streamlit app not found at {streamlit_app}")
        return 1
    
    # Configure environment for optimal performance
    env = os.environ.copy()
    env.update({
        'STREAMLIT_SERVER_ENABLE_STATIC_SERVING': 'true',
        'STREAMLIT_SERVER_ENABLE_CORS': 'false',
        'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
        'STREAMLIT_GLOBAL_DEVELOPMENT_MODE': 'false'
    })
    
    # Schedule browser opening
    open_browser(url, delay=2)
    
    print(f"🌐 Starting web server on port {port}...")
    print(f"📱 Access URL: {url}")
    print("⏹️  Press Ctrl+C to stop")
    
    try:
        # Launch Streamlit with optimized settings using global python3
        result = subprocess.run([
            'python3', '-m', 'streamlit', 'run', str(streamlit_app),
            '--server.port', str(port),
            '--server.address', 'localhost',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false',
            '--global.developmentMode', 'false',
            '--server.enableCORS', 'false',
            '--server.enableXsrfProtection', 'false',
            '--server.enableStaticServing', 'true',
            '--server.maxUploadSize', '200',
            '--theme.base', 'light'
        ], env=env, check=False)
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
