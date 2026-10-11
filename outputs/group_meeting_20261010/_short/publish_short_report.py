#!/usr/bin/env python3
"""Publish the reviewed six-page candidate to the user-authorized original path."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

SHORT=Path(__file__).resolve().parent
OUT=SHORT.parent
sys.path.insert(0,str(OUT/"_deps"))
import fitz

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    qpath=SHORT/"short_qa.json"
    qa=json.loads(qpath.read_text())
    review=(SHORT/"independent_review.md").read_text()
    assert "最终结论：PASS" in review,"Independent final audit has not passed"
    assert qa["pages"]==6 and qa["null_glyph_count"]==0
    assert not qa["literal_code_subscripts"] and not qa["out_of_page_text"]
    assert qa["raster_image_count"]==0 and all(qa["core_source_hashes_unchanged"].values())
    candidate=SHORT/"short_candidate.pdf"
    assert sha(candidate) in review,"Candidate changed after independent audit"
    target=OUT/"S4M_组会汇报_20261010.pdf"
    old_sha=sha(target)
    d=fitz.open(candidate)
    assert len(d)==6 and d[3].get_text().find("0.96%")>=0
    d.close()
    # Keep the candidate as source evidence; atomic rename only affects the
    # exact PDF explicitly selected by the user for replacement.
    temp=OUT/".short_publish.tmp.pdf"
    shutil.copyfile(candidate,temp)
    assert sha(temp)==sha(candidate)
    temp.replace(target)
    shutil.copyfile(SHORT/"五分钟口语稿.md",OUT/"组会口语稿.md")
    d=fitz.open(target)
    assert len(d)==6 and sha(target)==sha(candidate)
    qa.update({"review_status":"FINAL_PASS","root_visual_review":"all six candidate pages inspected; final regenerated diagram pages reinspected","independent_review":"independent_review.md","published_pdf":str(target),"published_pdf_sha256":sha(target),"previous_pdf_sha256":old_sha,"user_authorized_overwrite":True,"published_page_count":len(d)})
    d.close()
    qpath.write_text(json.dumps(qa,ensure_ascii=False,indent=2))
    print(json.dumps({"published_pdf":str(target),"pages":6,"bytes":target.stat().st_size,"same_path_overwrite":True,"independent_review":"FINAL_PASS","model_changes":0},ensure_ascii=False))

if __name__=="__main__":main()
