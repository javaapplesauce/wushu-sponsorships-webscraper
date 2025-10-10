import subprocess
import sys

# script paths
SCRAPER = "webscraper.py"
PERSONALIZER = "personalize_emails.py"
MAILER = "send_emails.py"

def run_cmd(cmd):
    """Run a command and stream its output live."""
    print(f"\n Running: {' '.join(cmd)}\n")
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.communicate()
    if process.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        sys.exit(process.returncode)
    else:
        print(f"Finished: {' '.join(cmd)}")

def main():
    # webscraper.py
    run_cmd([
        sys.executable, SCRAPER,
        "--location", "Columbia University, NY",
        "--radius", "5000",
        "--categories", "boba, asian_specialty, asian tea, matcha, skincare, asian market, asian_skincare",
        "--out", "results_nyc"
    ])

    # personalize_emails.py
    run_cmd([
        sys.executable, PERSONALIZER,
        "--csv", "results_nyc/shops.csv",
        "--template", "sponsorship_template.txt",
        "--event-location", "Columbia University, New York, NY",
        "--event-date", "10/25",
        "--event-time", "3:00 PM",
        "--event-place", "Columbia Sundial",
        "--out", "personalized_emails"
    ])

    # send_emails.py
    run_cmd([sys.executable, MAILER])

if __name__ == "__main__":
    main()
