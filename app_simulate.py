

import csv
import io
from flask import Blueprint, render_template, request, redirect, url_for

simulate_bp = Blueprint('simulate', __name__, url_prefix='/admin')

report_data = {
    "summary"      : [],
    "history"      : [],
    "weak_topics"  : [],
    "weak_material": [],
    "course_recs"  : [],
}


def _float(val):
    try:    return round(float(val), 4)
    except: return 0.0

def _int(val):
    try:    return int(float(val))
    except: return 0



def parse_csv(file_bytes: bytes) -> dict:
    text   = file_bytes.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows   = list(reader)

    if not rows:
        return report_data.copy()

    
    history = [
        {
            "q"     : r.get("question", ""),
            "score" : _float(r.get("final_score", 0)),
            "reward": _float(r.get("reward", 0)),
        }
        for r in rows
    ]

   
    last_per_topic = {}
    for r in rows:
        last_per_topic[r["topic"]] = r

    summary     = []
    weak_topics = []

    for topic, r in last_per_topic.items():
        avg_score = _float(r.get("topic_avg_score_so_far", 0))
        attempts  = _int(r.get("topic_attempts_so_far", 0))
        mastered  = r.get("is_mastered", "False").strip().upper() == "TRUE"
        prereqs   = r.get("topic_prereqs", "").strip()

        summary.append({
            "topic"    : topic,
            "attempts" : attempts,
            "avg_score": avg_score,
            "mastered" : mastered,
        })

        if attempts > 0 and avg_score < 0.5:
            weak_topics.append(topic)
        elif attempts == 0 and prereqs:
            weak_topics.append(topic)

    
    weak_material = []
    course_recs   = []

    try:
        from NLP.recommend_material import get_weak_topic_material, recommend_courses

        if weak_topics:
            weak_material = get_weak_topic_material(weak_topics)
            print(f"[recommend] weak_material: {weak_material}")
            course_recs = recommend_courses(weak_topics, top_n=5)
            print(f"[recommend] course_recs: {course_recs}")
        else:
            print("[recommend] No weak topics — skipping material/course lookup.")

    except Exception as e:
        print(f"[recommend] Could not load recommend_material: {e}")
        import traceback
        traceback.print_exc()

    return {
        "summary"      : summary,
        "history"      : history,
        "weak_topics"  : weak_topics,
        "weak_material": weak_material,
        "course_recs"  : course_recs,
    }



@simulate_bp.route("/")
def home():
    return render_template("upload.html")


@simulate_bp.route("/upload", methods=["POST"])
def upload():
    global report_data

    if "csv_file" not in request.files:
        return "No file uploaded.", 400

    f = request.files["csv_file"]
    if not f.filename.endswith(".csv"):
        return "Please upload a .csv file.", 400

    report_data = parse_csv(f.read())
    return redirect(url_for("simulate.report"))


@simulate_bp.route("/report")
def report():
    return render_template(
        "report.html",
        summary      = report_data["summary"],
        history      = report_data["history"],
        weak_topics  = report_data["weak_topics"],
        weak_material= report_data["weak_material"],
        course_recs  = report_data["course_recs"],
    )