#!/usr/bin/env python3
"""
Performance test script for Paokinator ML optimizations.
This script demonstrates the dramatic performance improvements.
"""

import time
import requests
import json
from typing import Dict, Any

class PerformanceTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
        
    def start_session(self) -> str:
        """Start a new game session."""
        print("🚀 Starting new game session...")
        start_time = time.time()
        
        response = requests.post(f"{self.base_url}/start")
        if response.status_code == 200:
            self.session_id = response.json()["session_id"]
            elapsed = (time.time() - start_time) * 1000
            print(f"✅ Session started in {elapsed:.2f}ms")
            return self.session_id
        else:
            raise Exception(f"Failed to start session: {response.status_code}")
    
    def get_question(self) -> Dict[str, Any]:
        """Get the next question and measure performance."""
        if not self.session_id:
            raise Exception("No active session")
        
        print(f"\n❓ Getting question for session {self.session_id[:8]}...")
        start_time = time.time()
        
        response = requests.get(f"{self.base_url}/question/{self.session_id}")
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print(f"⚡ Question served in {elapsed:.2f}ms")
            print(f"   Question: {data.get('question', 'N/A')}")
            print(f"   Feature: {data.get('feature', 'N/A')}")
            print(f"   Question #: {data.get('question_number', 'N/A')}")
            return data
        else:
            raise Exception(f"Failed to get question: {response.status_code}")
    
    def submit_answer(self, feature: str, answer: str) -> Dict[str, Any]:
        """Submit an answer and measure performance."""
        if not self.session_id:
            raise Exception("No active session")
        
        print(f"\n💬 Submitting answer '{answer}' for feature '{feature}'...")
        start_time = time.time()
        
        payload = {"feature": feature, "answer": answer}
        response = requests.post(f"{self.base_url}/answer/{self.session_id}", json=payload)
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print(f"⚡ Answer processed in {elapsed:.2f}ms")
            print(f"   Status: {data.get('status', 'N/A')}")
            return data
        else:
            raise Exception(f"Failed to submit answer: {response.status_code}")
    
    def get_predictions(self) -> Dict[str, Any]:
        """Get top predictions and measure performance."""
        if not self.session_id:
            raise Exception("No active session")
        
        print(f"\n🔮 Getting predictions...")
        start_time = time.time()
        
        response = requests.get(f"{self.base_url}/predictions/{self.session_id}")
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print(f"⚡ Predictions served in {elapsed:.2f}ms")
            predictions = data.get('top_predictions', [])
            for i, pred in enumerate(predictions[:3]):
                print(f"   {i+1}. {pred.get('animal', 'N/A')} ({pred.get('probability', 0):.3f})")
            return data
        else:
            raise Exception(f"Failed to get predictions: {response.status_code}")
    
    def run_performance_test(self):
        """Run a comprehensive performance test."""
        print("=" * 60)
        print("🔥 POKINATOR ML PERFORMANCE TEST")
        print("=" * 60)
        
        try:
            # Test 1: Session Creation
            self.start_session()
            
            # Test 2: First Question (Q0) - Should be INSTANT
            print("\n" + "=" * 40)
            print("TEST 1: Q0 Question (Should be < 50ms)")
            print("=" * 40)
            q0_data = self.get_question()
            
            # Test 3: Submit Answer
            if q0_data.get('feature') and q0_data.get('question'):
                feature = q0_data['feature']
                print(f"\n" + "=" * 40)
                print("TEST 2: Answer Processing")
                print("=" * 40)
                self.submit_answer(feature, "yes")
            
            # Test 4: Second Question (Q1) - Should be FAST
            print(f"\n" + "=" * 40)
            print("TEST 3: Q1 Question (Should be < 100ms)")
            print("=" * 40)
            q1_data = self.get_question()
            
            # Test 5: Get Predictions
            print(f"\n" + "=" * 40)
            print("TEST 4: Predictions (Should be < 50ms)")
            print("=" * 40)
            self.get_predictions()
            
            # Test 6: Performance Summary
            print(f"\n" + "=" * 40)
            print("TEST 5: Performance Summary")
            print("=" * 40)
            self.get_performance_metrics()
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    def get_performance_metrics(self):
        """Get server performance metrics."""
        try:
            response = requests.get(f"{self.base_url}/performance")
            if response.status_code == 200:
                data = response.json()
                print("📊 Server Performance Metrics:")
                print(f"   Question Cache Enabled: {data.get('optimizations', {}).get('question_cache_enabled', False)}")
                print(f"   Uniform Prior Cached: {data.get('optimizations', {}).get('uniform_prior_cached', False)}")
                print(f"   Precomputed Features: {data.get('optimizations', {}).get('precomputed_features', 0)}")
                print(f"   Prediction Cache Size: {data.get('optimizations', {}).get('prediction_cache_size', 0)}")
                
                print("\n💡 Performance Tips:")
                for tip in data.get('performance_tips', []):
                    print(f"   • {tip}")
            else:
                print("❌ Failed to get performance metrics")
        except Exception as e:
            print(f"❌ Error getting performance metrics: {e}")

def main():
    """Main test function."""
    print("Starting Paokinator ML Performance Test...")
    print("Make sure the server is running on http://localhost:8000")
    print()
    
    tester = PerformanceTester()
    
    try:
        tester.run_performance_test()
        print("\n" + "=" * 60)
        print("✅ PERFORMANCE TEST COMPLETED")
        print("=" * 60)
        print("\nExpected Results:")
        print("• Q0 Question: < 50ms (precomputed cache)")
        print("• Q1+ Questions: < 100ms (optimized calculations)")
        print("• Answer Processing: < 50ms (efficient updates)")
        print("• Predictions: < 50ms (cached results)")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")

if __name__ == "__main__":
    main()
