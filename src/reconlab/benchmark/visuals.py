"""Offline 3D point-cloud/camera inspection and posthoc pose measurements."""

import json
from pathlib import Path

import numpy as np

from .core import save_json, similarity_alignment


def export_geometry(geometry, source: Path, output: Path):
    data = json.loads((source / "transforms.json").read_text())
    frames = {Path(f["file_path"]).name: f for f in data["frames"]}
    gt = np.stack([frames[name]["transform_matrix"] for name in geometry.names]).astype(float)
    gt[:, :3, 1:3] *= -1
    scale, rotation, translation = similarity_alignment(geometry.c2w[:, :3, 3], gt[:, :3, 3])
    centers = scale * geometry.c2w[:, :3, 3] @ rotation.T + translation
    orientations = np.einsum("ij,njk->nik", rotation, geometry.c2w[:, :3, :3])
    delta = np.einsum("nij,nkj->nik", orientations, gt[:, :3, :3])
    angles = np.rad2deg(np.arccos(np.clip((np.trace(delta, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    save_json(
        output / "pose_metrics.json",
        {
            "alignment": "posthoc Sim3 fitted only on selected registered train camera centers",
            "rotation_median_deg": float(np.median(angles)),
            "rotation_mean_deg": float(np.mean(angles)),
            "center_rmse": float(np.sqrt(np.mean((centers - gt[:, :3, 3]) ** 2))),
            "registered": len(geometry.names),
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
    )
    points = scale * geometry.points @ rotation.T + translation
    finite = np.isfinite(points).all(1)
    points, colors = points[finite], geometry.colors[finite]
    if len(points) > 12000:
        selected = np.random.default_rng(42).choice(len(points), 12000, replace=False)
        points, colors = points[selected], colors[selected]
    # Viewer display clipping is disclosed; complete geometry.npz remains intact.
    center = np.median(points, axis=0) if len(points) else centers.mean(0)
    radius = max(
        float(np.quantile(np.linalg.norm(points - center, axis=1), 0.95)) if len(points) else 1.0,
        0.1,
    )
    display = np.linalg.norm(points - center, axis=1) < radius * 10
    payload = {
        "points": np.round((points[display] - center) / radius, 5).tolist(),
        "colors": colors[display].tolist(),
        "cameras": np.round((centers - center) / radius, 5).tolist(),
        "names": geometry.names,
    }
    html = """<!doctype html><meta charset="utf-8"><title>SfM reconstruction</title>
<style>body{margin:0;background:#f5f6f8;font:15px sans-serif}p{margin:12px}canvas{width:100vw;height:85vh}</style>
<p>Estimated SfM point cloud and camera centers · drag to rotate, wheel to zoom. GT used only for display alignment.
Up to 12,000 points shown; extreme display outliers hidden. Full geometry is in geometry.npz.</p><canvas id="c"></canvas>
<script>const D=PAYLOAD;const c=document.getElementById('c'),ctx=c.getContext('2d');let yaw=.4,pitch=.3,zoom=1,drag=null;
function project(p){let x=p[0]*Math.cos(yaw)-p[2]*Math.sin(yaw),z=p[0]*Math.sin(yaw)+p[2]*Math.cos(yaw);let y=p[1]*Math.cos(pitch)-z*Math.sin(pitch);return[x,y,z*Math.cos(pitch)+p[1]*Math.sin(pitch)];}
function draw(){c.width=c.clientWidth;c.height=c.clientHeight;ctx.fillStyle='#fff';ctx.fillRect(0,0,c.width,c.height);let s=Math.min(c.width,c.height)*.17*zoom;
let pts=D.points.map((p,i)=>[...project(p),i]).sort((a,b)=>a[2]-b[2]);for(let p of pts){let col=D.colors[p[3]];ctx.fillStyle=`rgb(${col.join(',')})`;ctx.fillRect(c.width/2+p[0]*s,c.height/2-p[1]*s,2,2);}
ctx.strokeStyle='#1768bd';ctx.fillStyle='#1768bd';ctx.beginPath();D.cameras.forEach((p,i)=>{let q=project(p),x=c.width/2+q[0]*s,y=c.height/2-q[1]*s;if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);ctx.fillRect(x-3,y-3,6,6);});ctx.stroke();}
c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=null;c.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];draw()}};c.onwheel=e=>{e.preventDefault();zoom*=Math.exp(-e.deltaY*.001);draw()};window.onresize=draw;draw();</script>"""
    (output / "geometry.html").write_text(
        html.replace("PAYLOAD", json.dumps(payload, separators=(",", ":")))
    )
