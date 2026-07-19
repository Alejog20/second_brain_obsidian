from typing import Dict, List, Any
import datetime

class ReportGenerator:
    """
    Generates the morning report summarizing the night's work.
    Requirement 5: nightly change report.
    """
    def __init__(self):
        pass

    def generate_report(self, stats: Dict[str, Any], items: List[Dict[str, Any]], flags: List[Any]) -> str:
        """
        Constructs the final markdown report string.
        
        Args:
            stats: Dictionary containing summary metrics (count of scans, changes, new notes).
            items: A list of specific significant changes/actions taken.
            flags: A list of items flagged for manual review (e.g., low clarity).
            
        Returns:
            A formatted markdown string for the report file.
        """
        date_str = datetime.datetime.now().strftime("%m-%d-%Y")
        report_title = f"# Review — {date_str}"
        
        # 1. Summary Header
        summary_line = f"Run: {stats.get('start_time')}–{stats.get('end_time')} · " \
                        f"{stats['scanned']} notes scanned · " \
                        f"{stats['changed']} changed · " \
                        f"{stats['new']} new · " \
                        f"est. cost ${stats.get('cost', '0.00')}"
        
        report = [report_title, "", summary_line, ""]

        # 2. Significant Changes
        report.append("## Significant changes")
        if not items:
            report.append("(No major structural changes detected.)")
        else:
            for item in items:
                # Item format: ["Reason", "Description/Action"]
                reason = item.get("reason", "Update")
                detail = item.get("detail", "")
                report.append(f"### {reason}")
                report.append(f"{detail}\n")

        # 3. New notes from today's daily digestion
        report.append("## New notes from today's daily digestion")
        if not flags: # Or use a specific list for new items if separate
            report.append("No new notes generated today.")
        else:
            for f in flags:
                # Just listing the notes that were successfully distilled
                report.append(f"- [{f}]")
        
        # 4. Minor changes
        minor_count = stats.get('minor_changes', 0)
        report.append("## Minor changes")
        if minor_count > 0:
            report.append(f"{minor_count} total — see full log.")
            report.append(f"[View full diff log](_reports/{date_str}-full-diff.json)")
        else:
            report.append("No minor changes recorded.")

        # 5. Flagged for review
        report.append("## Flagged for review")
        if not flags:
            report.append("All items successfully processed.")
        else:
            for f in flags:
                # Note check logic (e.g., low clarity)
                reason = "clarity score low" if "low_clarity" in str(f).lower() else "needs review"
                report.append(f"- [{f}] — {reason}")

        return "\n".join(report)

report_generator = ReportGenerator()

def generate_morning_report(stats: Dict[str, Any], items: List[Dict[str, Any]], flags: List[Any]) -> str:
    return report_generator.generate_report(stats, items, flags)