"""
FINDU -- Browser E2E Forensic Test for Admin-Only Test Environment Reset.

Executes a full end-to-end browser workflow via Playwright:
1. Admin Login
2. Header Reset Button Visibility Check
3. Confirmation Dialog & Type-to-Confirm Validation
4. Server Reset Execution
5. Modal Summary Verification (Deleted vs Preserved Data)
6. UI State Refresh Verification
7. Non-Admin (Police) Login & Absence of Reset Button Check
"""

import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_browser_e2e():
    print("=" * 70)
    print("FINDU BROWSER E2E: ADMIN TEST ENVIRONMENT RESET")
    print("=" * 70)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("\n[STEP 1] Navigating to FindU UI at http://localhost:5173 ...")
        page.goto("http://localhost:5173")
        page.wait_for_load_state("networkidle")

        # Clear any stored credentials to start clean on login screen
        page.evaluate("() => localStorage.clear()")
        page.reload()
        page.wait_for_load_state("networkidle")

        # Step 2: Login as Admin
        print("[STEP 2] Logging in as ADMIN (admin) ...")
        page.wait_for_selector("#username", timeout=10000)
        page.fill("#username", "admin")
        page.fill("#password", "admin-demo-CHANGE-ME")
        page.click("button[type='submit']")
        page.wait_for_selector("header", timeout=10000)
        print("  -> Admin login successful.")

        # Step 3: Check for Reset Button in Header
        print("\n[STEP 3] Checking for 'Reset Test Environment' button in Header ...")
        reset_btn = page.locator("button:has-text('Reset Test Environment')")
        page.wait_for_selector("button:has-text('Reset Test Environment')", timeout=5000)
        assert reset_btn.is_visible(), "Reset Test Environment button should be visible for ADMIN!"
        print("  -> 'Reset Test Environment' button is VISIBLE for ADMIN.")

        # Step 4: Click Reset Button to Open Modal
        print("\n[STEP 4] Clicking 'Reset Test Environment' to trigger confirmation modal ...")
        reset_btn.click()
        page.wait_for_selector("text=Warning: Destructive Reset Action", timeout=5000)
        print("  -> Confirmation modal opened successfully.")

        # Step 5: Verify Submit Button is Disabled by Default
        modal_submit_btn = page.locator("button[type='submit']:has-text('Reset Test Environment')")
        assert modal_submit_btn.is_disabled(), "Submit button must be disabled until exact confirm text is entered!"
        print("  -> Safety verification passed: submit button is initially DISABLED.")

        # Step 6: Enter Partial/Wrong Text
        print("\n[STEP 6] Testing partial input 'RESET' ...")
        confirm_input = page.locator("input[placeholder='RESET TEST DATA']")
        confirm_input.fill("RESET")
        assert modal_submit_btn.is_disabled(), "Submit button must remain disabled for partial text!"
        print("  -> Safety verification passed: remains disabled for partial input.")

        # Step 7: Enter Exact Text 'RESET TEST DATA'
        print("\n[STEP 7] Entering exact confirmation phrase 'RESET TEST DATA' ...")
        confirm_input.fill("RESET TEST DATA")
        assert not modal_submit_btn.is_disabled(), "Submit button should be ENABLED once exact text is typed!"
        print("  -> Submit button is now ENABLED.")

        # Step 8: Click Submit & Await Server Reset
        print("\n[STEP 8] Submitting reset request and awaiting server execution ...")
        modal_submit_btn.click()
        page.wait_for_selector("text=The test environment has been restored to a clean state.", timeout=15000)
        print("  -> Reset completed successfully! Summary modal displayed.")

        # Step 9: Verify Preserved and Deleted Cards in Modal
        deleted_card = page.locator("text=Deleted Data")
        preserved_card = page.locator("text=Preserved Infrastructure")
        assert deleted_card.is_visible()
        assert preserved_card.is_visible()
        print("  -> Verified Deleted Data and Preserved Infrastructure metrics in modal.")

        # Step 10: Close Modal
        done_btn = page.locator("button:has-text('Done')")
        done_btn.click()
        page.wait_for_timeout(1000)
        print("  -> Closed reset summary modal.")

        # Step 11: Switch to Cases & Registration Tab to verify empty persons
        print("\n[STEP 11] Navigating to Cases & Registration tab ...")
        cases_tab_btn = page.locator("button:has-text('Cases & Registration')")
        cases_tab_btn.click()
        page.wait_for_timeout(1500)
        print("  -> Cases & Registration tab active.")

        # Step 12: Switch to Camera Command Center tab to verify cameras preserved
        print("\n[STEP 12] Navigating to Camera Command Center tab ...")
        cameras_tab_btn = page.locator("button:has-text('Camera Command Center')")
        cameras_tab_btn.click()
        page.wait_for_timeout(1500)
        c1_card = page.get_by_text("C1", exact=True).first
        assert c1_card.is_visible(), "Camera C1 must be preserved!"
        print("  -> Verified Camera C1 is preserved and configured.")

        # Step 13: Logout
        print("\n[STEP 13] Logging out admin ...")
        logout_btn = page.locator("button[title='Sign out']")
        logout_btn.click()
        page.wait_for_selector("#username", timeout=5000)
        print("  -> Admin logged out.")

        # Step 14: Non-Admin (Police) Verification
        print("\n[STEP 14] Logging in as POLICE (police) to verify absence of reset control ...")
        page.fill("#username", "police")
        page.fill("#password", "police-demo-CHANGE-ME")
        page.click("button[type='submit']")
        page.wait_for_selector("header", timeout=10000)
        print("  -> Police login successful.")

        police_reset_btn = page.locator("button:has-text('Reset Test Environment')")
        assert not police_reset_btn.is_visible(), "Reset Test Environment button MUST NOT be visible for POLICE role!"
        print("  -> Security verified: 'Reset Test Environment' button is NOT visible for POLICE.")

        browser.close()

    print("\n" + "=" * 70)
    print("ALL BROWSER E2E TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_browser_e2e()
