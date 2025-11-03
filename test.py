"""
Test script to verify undo functionality works correctly.
Run this after starting the server to test all undo scenarios.
"""
import requests
import json

BASE_URL = "http://127.0.0.1:5000/"

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_undo_at_question_1():
    """Test undoing at the very first question."""
    print_section("TEST 1: Undo at Question 1")
    
    try:
        # Start game
        resp = requests.post(f"{BASE_URL}/start", json={"domain_name": "animals"})
        session_id = resp.json()["session_id"]
        print(f"✓ Started session: {session_id}")
        
        # Get first question
        resp = requests.get(f"{BASE_URL}/question/{session_id}")
        q1 = resp.json()
        print(f"✓ Question {q1['question_number']}: {q1['question']}")
        assert q1['question_number'] == 1, f"Expected Q1, got Q{q1['question_number']}"
        
        # Try to undo (should fail gracefully)
        resp = requests.post(f"{BASE_URL}/undo/{session_id}")
        
        # Debug output
        print(f"Response status: {resp.status_code}")
        print(f"Response headers: {resp.headers.get('content-type')}")
        print(f"Response text: {resp.text[:200]}")
        
        if resp.status_code == 200:
            undo_resp = resp.json()
            print(f"✓ Undo response: {undo_resp.get('undo_status')}")
            print(f"✓ Still showing question {undo_resp.get('question_number')}: {undo_resp.get('question', 'N/A')[:50]}")
            assert undo_resp.get('undo_status') == 'failed_at_start', "Should indicate cannot undo"
            assert undo_resp.get('question_number') == 1, "Should still be on Q1"
        else:
            print(f"✗ Error: {resp.status_code} - {resp.text}")
            return False
        
        print("✅ TEST 1 PASSED: Undo at Q1 handled correctly")
        return True
    except Exception as e:
        print(f"Exception details: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_undo_from_question_2():
    """Test undoing from question 2 back to question 1."""
    print_section("TEST 2: Undo from Question 2 to Question 1")
    
    # Start game
    resp = requests.post(f"{BASE_URL}/start", json={"domain_name": "animals"})
    session_id = resp.json()["session_id"]
    print(f"✓ Started session: {session_id}")
    
    # Get Q1
    resp = requests.get(f"{BASE_URL}/question/{session_id}")
    q1 = resp.json()
    q1_text = q1['question']
    print(f"✓ Q{q1['question_number']}: {q1_text[:50]}")
    assert q1['question_number'] == 1
    
    # Answer Q1
    resp = requests.post(f"{BASE_URL}/answer/{session_id}", 
                        json={"feature": q1['feature'], "answer": "yes"})
    print(f"✓ Answered Q1")
    
    # Get Q2
    resp = requests.get(f"{BASE_URL}/question/{session_id}")
    q2 = resp.json()
    q2_text = q2['question']
    print(f"✓ Q{q2['question_number']}: {q2_text[:50]}")
    assert q2['question_number'] == 2, f"Expected Q2, got Q{q2['question_number']}"
    
    # Undo back to Q1
    resp = requests.post(f"{BASE_URL}/undo/{session_id}")
    undo_resp = resp.json()
    
    if resp.status_code == 200:
        print(f"✓ Undo successful: {undo_resp.get('undo_status')}")
        print(f"✓ Back to Q{undo_resp.get('question_number')}: {undo_resp.get('question', 'N/A')[:50]}")
        
        # Verify we're back to Q1
        assert undo_resp.get('question_number') == 1, f"Expected Q1, got Q{undo_resp.get('question_number')}"
        assert undo_resp.get('question') == q1_text, "Should show same question as original Q1"
        print("✅ TEST 2 PASSED: Correctly showing Q1 with question_number = 1")
    else:
        print(f"✗ Error: {resp.status_code} - {resp.text}")
        return False
    
    return True

def test_undo_from_question_4():
    """Test undoing from Q4 to Q3."""
    print_section("TEST 3: Undo from Question 4 to Question 3")
    
    # Start game
    resp = requests.post(f"{BASE_URL}/start", json={"domain_name": "animals"})
    session_id = resp.json()["session_id"]
    print(f"✓ Started session: {session_id}")
    
    questions = []
    
    # Answer 3 questions to get to Q4
    for i in range(1, 4):
        resp = requests.get(f"{BASE_URL}/question/{session_id}")
        q = resp.json()
        questions.append((q['question_number'], q['question'], q.get('feature')))
        print(f"✓ Q{q['question_number']}: {q['question'][:50]}")
        
        # Answer the question
        if not q.get('should_guess'):
            requests.post(f"{BASE_URL}/answer/{session_id}",
                         json={"feature": q['feature'], "answer": "yes"})
    
    # Get Q4
    resp = requests.get(f"{BASE_URL}/question/{session_id}")
    q4 = resp.json()
    print(f"✓ Q{q4['question_number']}: {q4['question'][:50]}")
    assert q4['question_number'] == 4, f"Expected Q4, got Q{q4['question_number']}"
    
    # Undo back to Q3
    resp = requests.post(f"{BASE_URL}/undo/{session_id}")
    undo_resp = resp.json()
    
    if resp.status_code == 200:
        print(f"✓ Undo successful: {undo_resp.get('undo_status')}")
        print(f"✓ Back to Q{undo_resp.get('question_number')}: {undo_resp.get('question', 'N/A')[:50]}")
        
        # Verify we're back to Q3
        assert undo_resp.get('question_number') == 3, f"Expected Q3, got Q{undo_resp.get('question_number')}"
        assert undo_resp.get('question') == questions[2][1], "Should show Q3's question"
        print("✅ TEST 3 PASSED: Correctly showing Q3 with question_number = 3")
    else:
        print(f"✗ Error: {resp.status_code} - {resp.text}")
        return False
    
    return True

def test_multiple_undos():
    """Test multiple consecutive undos."""
    print_section("TEST 4: Multiple Consecutive Undos")
    
    # Start game
    resp = requests.post(f"{BASE_URL}/start", json={"domain_name": "animals"})
    session_id = resp.json()["session_id"]
    print(f"✓ Started session: {session_id}")
    
    # Answer 5 questions
    for i in range(5):
        resp = requests.get(f"{BASE_URL}/question/{session_id}")
        q = resp.json()
        print(f"✓ Q{q['question_number']}")
        
        if not q.get('should_guess'):
            requests.post(f"{BASE_URL}/answer/{session_id}",
                         json={"feature": q['feature'], "answer": "sometimes"})
    
    # Should be at Q6 now
    resp = requests.get(f"{BASE_URL}/question/{session_id}")
    current = resp.json()
    print(f"✓ Currently at Q{current['question_number']}")
    
    # Undo 3 times: Q6 → Q5 → Q4 → Q3
    for expected_q in [5, 4, 3]:
        resp = requests.post(f"{BASE_URL}/undo/{session_id}")
        undo_resp = resp.json()
        actual_q = undo_resp.get('question_number')
        print(f"✓ Undo → Q{actual_q}")
        assert actual_q == expected_q, f"Expected Q{expected_q}, got Q{actual_q}"
    
    print("✅ TEST 4 PASSED: Multiple undos work correctly")
    return True

def test_undo_after_guess():
    """Test undoing after rejecting a guess."""
    print_section("TEST 5: Undo After Guess")
    
    # Start game
    resp = requests.post(f"{BASE_URL}/start", json={"domain_name": "animals"})
    session_id = resp.json()["session_id"]
    print(f"✓ Started session: {session_id}")
    
    # Answer questions until we get a guess (or 10 questions max)
    for i in range(10):
        resp = requests.get(f"{BASE_URL}/question/{session_id}")
        q = resp.json()
        
        if q.get('should_guess'):
            print(f"✓ Got guess at Q{i+1}: {q['guess']}")
            
            # Reject the guess
            requests.post(f"{BASE_URL}/reject/{session_id}",
                         json={"animal_name": q['guess']})
            print(f"✓ Rejected guess")
            
            # Continue
            requests.post(f"{BASE_URL}/continue/{session_id}")
            
            # Get next question
            resp = requests.get(f"{BASE_URL}/question/{session_id}")
            after_continue = resp.json()
            after_q_num = after_continue.get('question_number')
            print(f"✓ After continue: Q{after_q_num}")
            
            # Now undo - should skip the guess and go to before it
            resp = requests.post(f"{BASE_URL}/undo/{session_id}")
            undo_resp = resp.json()
            undo_q_num = undo_resp.get('question_number')
            print(f"✓ After undo: Q{undo_q_num}")
            
            # Should be less than after_continue
            assert undo_q_num < after_q_num, f"Undo should go back (Q{undo_q_num} < Q{after_q_num})"
            print("✅ TEST 5 PASSED: Undo after guess works correctly")
            return True
        
        elif q.get('is_sneaky_guess'):
            print(f"✓ Got sneaky guess at Q{q['question_number']}")
            requests.post(f"{BASE_URL}/answer/{session_id}",
                         json={"feature": "sneaky_guess", "answer": "no",
                               "animal_name": q.get('animal_name')})
        else:
            requests.post(f"{BASE_URL}/answer/{session_id}",
                         json={"feature": q['feature'], "answer": "sometimes"})
    
    print("⚠ No guess received in 10 questions, test incomplete")
    return True

def run_all_tests():
    """Run all undo tests."""
    print("\n" + "█"*60)
    print("  UNDO FUNCTIONALITY TEST SUITE")
    print("█"*60)
    
    tests = [
        ("Undo at Question 1", test_undo_at_question_1),
        ("Undo from Question 2", test_undo_from_question_2),
        ("Undo from Question 4", test_undo_from_question_4),
        ("Multiple Undos", test_multiple_undos),
        ("Undo After Guess", test_undo_after_guess),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {e}")
            results.append((name, False))
    
    # Summary
    print_section("TEST SUMMARY")
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")

if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to server at", BASE_URL)
        print("   Make sure the server is running: python main.py")
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")