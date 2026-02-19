# app/services/scheduler_service.py
from apscheduler.schedulers.background import BackgroundScheduler
from app.supabase_client import get_supabase
from app.services.revalidate_service import revalidate_attendance

REVALIDATE_INTERVAL_MINUTES = 30

scheduler = BackgroundScheduler()

def auto_revalidate_job():
    print("Running background revalidation job...")

    sb = get_supabase()

    # Find provisional attendance rows
    rows = (
        sb.table("attendance")
        .select("id,status")
        .in_("status", ["present_soft", "suspicious"])
        .execute()
    )

    if not rows.data:
        print("No rows need revalidation")
        return

    for row in rows.data:
        try:
            print(f"Revalidating attendance id {row['id']}")
            revalidate_attendance(row["id"])
        except Exception as e:
            print("Revalidation failed:", e)

def start_scheduler():
    scheduler.add_job(
        auto_revalidate_job,
        "interval",
        minutes=REVALIDATE_INTERVAL_MINUTES,
        id="attendance_revalidator",
        replace_existing=True,
    )
    scheduler.start()
    print("Background scheduler started.")
