"""Build a timestamped, auditable assignment report from immutable completed artifacts.

CPU only. Never changes experiments, ranks on test data, or substitutes failed objects.
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, html, json, math, shutil, statistics
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path('/mnt/afs/260010098/code/3dv_assignment')
DATA = Path('/mnt/afs/260010098/data/3dv_assignment')
TASK = Path('/mnt/afs/260010098/data/coordination/tasks/20261117_assignment2_report_r01')
RUNTIME = Path('/mnt/afs/260010098/code/runner/20261117_roma100_filter_r04')
LABELS = {'baseline':'RoMa v2 原版','fg':'仅前景过滤','fg_conf050':'前景＋overlap≥0.5'}


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def avg(values):
    values=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.mean(values) if values else None


def fmt(value):
    return '—' if value is None else f'{value:.2f}' if isinstance(value,float) else str(value)


def table(headers,rows):
    def cell(x):return fmt(x).replace('|','／').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(cell(x) for x in row)+' |' for row in rows])


def json_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def submission_asset(relative):
    relative=Path(relative)
    if relative==Path('figures/filter_ablation.png'):return False
    if relative.name=='reconstruction.gif':return False
    if 'filter_audit' in relative.parts and relative.suffix=='.npz':return False
    if 'model_inputs' in relative.parts and relative.suffix=='.png':
        return relative.parent.name=='fine_1280' and relative.name=='view_001.png'
    return True


class EvidenceCollector:
    def __init__(self):
        self.config=read(RUNTIME/'config.json')
        self.objects={o['uid']:o for o in self.config['objects']}
        self.selection=read(TASK/'showcase_selection.json')
        self.pins={}

    def record(self,path):
        if path.exists():self.pins[str(path)]={'sha256':sha(path),'bytes':path.stat().st_size}
        return read(path)

    def case(self,folder,uid,method):
        status=self.record(folder/'status.json') or {'state':'PENDING'}
        g=self.record(folder/'geometry.json')
        if not g and method=='fg_conf050' and uid in self.config['selection_uids']:
            g=self.record(DATA/'roma100_filter_r01/comparison_geometry/fg_conf050'/uid/'geometry.json')
        result={'uid':uid,'name':self.objects[uid]['name'],'method':method,'state':status['state'],
                'path':str(folder),'registered':g.get('registered',status.get('registered')),'points':g.get('points3d'),
                'error':status.get('error'),'seconds':status.get('seconds'),'reused_selection_case':uid in self.config['selection_uids']}
        if status['state']=='COMPLETE':
            metric=self.record(folder/'gaussian/render_metrics.json').get('views',[])
            pose=self.record(folder/'gaussian/pose_metrics.json') or self.record(folder/'pose_metrics.json')
            result['rotation_deg']=pose.get('rotation_median_deg')
            for split in ['val','test']:
                for key in ['psnr','foreground_psnr','ssim']:
                    vals=[m.get(key) for m in metric if m['split']==split]
                    result[split+'_'+key]=avg(vals) if len(vals)==6 and all(x is not None for x in vals) else None
            result['training']=self.record(folder/'gaussian/training.json')
        return result

    def collect(self):
        start=datetime.datetime.now(datetime.timezone.utc).isoformat()
        original=DATA/'matching10_sparse6_r01'
        frontend=[]
        for method in sorted(p.name for p in (original/'results').iterdir() if p.is_dir()):
            frontend.extend(self.case(original/'results'/method/u,u,method) for u in self.config['selection_uids'])
        ablation=[]
        for method in LABELS:
            ablation.extend(self.case(DATA/'roma100_filter_r01/comparison'/method/u,u,method) for u in self.config['selection_uids'])
        current=Path(self.config['output'])/'results/fg_conf050'
        expansion=[self.case(current/o['uid'],o['uid'],'fg_conf050') for o in self.config['objects']]
        gt=[]
        for method in ['3dgs','3dgs_mcmc','2dgs','3dgut_mcmc']:
            for uid in self.config['selection_uids']:
                folder=DATA/'gaussian10_sparse6_r01/results'/method/uid
                s=self.record(folder/'status.json');tr=self.record(folder/'training.json')
                va=self.record(folder/'val_metrics.json');te=self.record(folder/'test_metrics.json')
                gt.append({'uid':uid,'method':method,'state':s.get('state','PENDING'),'steps':tr.get('steps'),
                           'best_step':tr.get('best_step'),'val_foreground_psnr':va.get('foreground_psnr') if s.get('state')=='COMPLETE' else None,
                           'test_foreground_psnr':te.get('foreground_psnr') if s.get('state')=='COMPLETE' else None,'seconds':tr.get('seconds')})
        views=[]
        for p in sorted((DATA/'matching10_r01/view_ablation').glob('*/*/*/status.json')):
            s=self.record(p);pose=self.record(p.parent/'pose_metrics.json')
            views.append({'method':p.parent.parent.parent.name,'uid':p.parent.parent.name,'subset':p.parent.name,
                          'state':s['state'],'input_views':s.get('input_views'),'registered':s.get('registered'),
                          'rotation_deg':pose.get('rotation_median_deg'),'path':str(p.parent)})
        return {'snapshot_started_utc':start,'snapshot_finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'config':self.config,'showcase_uids':self.selection['confirmed_uids'],'frontend':frontend,'ablation':ablation,
                'gt':gt,'views_historical':views,'expansion':expansion,'evidence_files':self.pins,
                'scope':'Current raw public Objaverse data, notAssignment1cleaneddata; pending is notfailure; user-selectedshowcase is illustrative'}


class ArtifactPublisher:
    def __init__(self,output):self.output=output;self.files={}
    def copy(self,source,relative):
        dst=self.output/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        expected=sha(source)
        if not dst.exists() or sha(dst)!=expected:shutil.copyfile(source,dst)
        assert sha(dst)==expected
        self.files[str(relative)]={'source':str(source),'sha256':expected,'bytes':dst.stat().st_size}
        return str(relative)
    def showcase(self,snapshot):
        rows=[]
        for index,uid in enumerate(snapshot['showcase_uids'],1):
            root=DATA/'roma100_filter_r01/results/fg_conf050'/uid
            assert read(root/'status.json').get('state')=='COMPLETE',uid
            base=Path('assets/showcase')/uid
            required=['matching.json','geometry.json','geometry.npz','status.json','gaussian/training.json',
                      'gaussian/render_metrics.json','gaussian/pose_metrics.json','gaussian/centered.gif',
                      'gaussian/centered.json','gaussian/render.json']
            for name in required:self.copy(root/name,base/name)
            for sub in ['matches','visuals','model_inputs','foreground_masks','filter_audit','raw_pair_provenance']:
                for p in sorted((root/sub).rglob('*')):
                    if p.is_file():self.copy(p,base/p.relative_to(root))
            # Source PNG and tensor metadata are preserved separately, without display re-encoding.
            inputs=DATA/'roma100_filter_r01/inputs'/uid
            for p in inputs.glob('images/*.png'):self.copy(p,base/'source_inputs'/p.name)
            self.copy(inputs/'input_manifest.json',base/'source_inputs/input_manifest.json')
            seq=root/'gaussian/centered';frames=sorted(seq.glob('frame_*.png'));assert len(frames)==72
            frame_hashes=[{'name':p.name,'sha256':sha(p)} for p in frames]
            json_write(self.output/base/'gaussian/png_sequence.json',{'source':str(seq),'frames':frame_hashes,'fps':24,'seconds':3,'size':[512,512]})
            self.copy(frames[0],base/'gaussian/still.png')
            with Image.open(root/'gaussian/centered.gif') as gif:
                ms=0
                for n in range(gif.n_frames):gif.seek(n);ms+=gif.info.get('duration',0)
                assert gif.size==(512,512) and ms==3000
                gif_frames=gif.n_frames
            matching=sorted((root/'visuals').glob('*.png'));assert matching
            row=next(x for x in snapshot['expansion'] if x['uid']==uid)
            item=next(x for x in snapshot['config']['objects'] if x['uid']==uid)
            rows.append({**row,'index':index,'license':item['license'],'source_url':item['url'],
                         'gif':str(base/'gaussian/centered.gif'),'still':str(base/'gaussian/still.png'),
                         'match':str(base/'visuals'/matching[0].name),'encoded_gif_frames':gif_frames,'gif_duration_ms':ms,
                         'source_view':str(base/'model_inputs/fine_1280/view_001.png'),'mask':str(base/'foreground_masks/view_001.png')})
        return rows


class AssignmentReportBuilder:
    def __init__(self,output):self.output=output;self.output.mkdir(parents=True,exist_ok=True)
    def plots(self,snapshot):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
        variants=list(LABELS);common=set.intersection(*[{r['uid'] for r in snapshot['ablation'] if r['method']==m and r.get('val_foreground_psnr') is not None} for m in variants])
        colors=['#55718a','#249b89','#d57c32'];labels=['Original','Foreground','Foreground + score >= 0.5']
        axes[0].bar(range(3),[sum(r['state']=='COMPLETE' for r in snapshot['ablation'] if r['method']==m) for m in variants],color=colors)
        axes[0].set(ylim=(0,11),ylabel='Completed objects / 10',xticks=range(3),xticklabels=labels);axes[0].tick_params(axis='x',labelrotation=12)
        for uid in sorted(common):
            vals=[next(r['val_foreground_psnr'] for r in snapshot['ablation'] if r['uid']==uid and r['method']==m) for m in variants]
            axes[1].plot(range(3),vals,'o-',alpha=.45,linewidth=1)
        means=[avg([r['val_foreground_psnr'] for r in snapshot['ablation'] if r['uid'] in common and r['method']==m]) for m in variants]
        axes[1].plot(range(3),means,'ks-',linewidth=2,label=f'Mean on same {len(common)} objects')
        axes[1].set(ylabel='Validation foreground PSNR (dB)',xticks=range(3),xticklabels=labels);axes[1].tick_params(axis='x',labelrotation=12);axes[1].legend(fontsize=8)
        (self.output/'assets/figures').mkdir(parents=True,exist_ok=True)
        fig.savefig(self.output/'assets/figures/filter_ablation.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(12,2.2));ax.set(xlim=(0,12),ylim=(0,2));ax.axis('off')
        stages=['6 RGB views\n512 x 512','RoMa v2\n15 image pairs','Foreground +\noverlap >= 0.5','RANSAC + SfM\nposes / points','3DGS\n3000 steps','Held-out views\n+ 72-frame GIF']
        for i,label in enumerate(stages):
            ax.text(i*2+1,1,label,ha='center',va='center',fontsize=10,bbox={'boxstyle':'round,pad=.6','facecolor':'#dff3ee','edgecolor':'#249b89'})
            if i<5:ax.annotate('',xy=(i*2+2.3,1),xytext=(i*2+1.7,1),arrowprops={'arrowstyle':'->','color':'#334155'})
        fig.savefig(self.output/'assets/figures/pipeline.png',bbox_inches='tight',dpi=180);plt.close(fig)
        return common,means
    def tables(self,s):
        core=s['config']['selection_uids'];result={}
        result['CORE_OBJECTS']=table(['序号','物体','UID'],[(i+1,next(o['name'] for o in s['config']['objects'] if o['uid']==u),u) for i,u in enumerate(core)])
        rows=[]
        for m in sorted({r['method'] for r in s['frontend']}):
            group=[r for r in s['frontend'] if r['method']==m];good=[r for r in group if r['state']=='COMPLETE']
            rows.append((m,f'{len(good)}/10',avg([r.get('val_foreground_psnr') for r in good]),avg([r.get('test_foreground_psnr') for r in good]),avg([r.get('rotation_deg') for r in good])))
        result['FRONTENDS']=table(['前端 / 完整系统','流程完成','验证前景PSNR↑','测试前景PSNR↑','逐物体旋转中位误差的均值↓（°）'],rows)
        common=set.intersection(*[{r['uid'] for r in s['ablation'] if r['method']==m and r.get('val_foreground_psnr') is not None} for m in LABELS])
        rows=[]
        for m,label in LABELS.items():
            good=[r for r in s['ablation'] if r['method']==m and r['state']=='COMPLETE']
            paired=[r for r in good if r['uid'] in common]
            rows.append((label,f'{len(good)}/10',avg([r.get('val_foreground_psnr') for r in good]),avg([r.get('val_foreground_psnr') for r in paired]),avg([r.get('test_foreground_psnr') for r in paired])))
        result['ABLATION']=table(['过滤方案','完成数','各自成功子集验证PSNR','共同9例验证PSNR','共同9例测试PSNR（后验）'],rows)
        rows=[]
        for uid in core:
            records=[next(r for r in s['ablation'] if r['uid']==uid and r['method']==m) for m in LABELS]
            rows.append([s['config']['objects'][next(i for i,o in enumerate(s['config']['objects']) if o['uid']==uid)]['name']]+[r.get('val_foreground_psnr') for r in records]+[f"{r.get('registered','—')}/6" for r in records])
        result['PER_OBJECT_ABLATION']=table(['原10例','原版Val','前景Val','前景＋分数Val','原版注册','前景注册','前景＋分数注册'],rows)
        rows=[]
        for m in ['3dgs','3dgs_mcmc','2dgs','3dgut_mcmc']:
            group=[r for r in s['gt'] if r['method']==m and r['state']=='COMPLETE']
            rows.append((m,f'{len(group)}/10',avg([r['val_foreground_psnr'] for r in group]),avg([r['test_foreground_psnr'] for r in group]),avg([r['seconds'] for r in group])))
        result['GT_TABLE']=table(['GT相机下的方法','完成数','验证前景PSNR↑','测试前景PSNR↑','平均训练秒数'],rows)
        groups=defaultdict(list)
        for r in s['views_historical']:groups[r['subset']].append(r)
        result['VIEW_TABLE']=table(['历史视图设置','已有记录','SfM流程完成','全部输入均注册','有效结果旋转误差均值（°）'],[(k,len(v),sum(r['state']=='COMPLETE' for r in v),sum(r['state']=='COMPLETE' and r['registered']==r['input_views'] for r in v),avg([r['rotation_deg'] for r in v if r['state']=='COMPLETE'])) for k,v in sorted(groups.items())])
        counts=Counter(r['state'] for r in s['expansion'])
        result['PROGRESS']=table(['统计范围','完成','失败','正在运行','尚未开始'],[(name,*[sum(r['state']==state for r in rows) for state in ['COMPLETE','FAILED','RUNNING','PENDING']]) for name,rows in [('固定100例',s['expansion']),('原10例对照',[r for r in s['expansion'] if r['reused_selection_case']]),('其余90例扩展',[r for r in s['expansion'] if not r['reused_selection_case']])]])
        result['TIMESTAMP']=s['snapshot_finished_utc'];result['SUCCESS_COUNT']=str(counts['COMPLETE'])
        return result
    def export_csv(self,name,rows):
        fields=[]
        for row in rows:
            fields.extend(k for k,v in row.items() if k not in fields and not isinstance(v,(dict,list)))
        target=self.output/'tables'/f'{name}.csv';target.parent.mkdir(exist_ok=True)
        with target.open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    def render(self,s,publisher,showcase):
        from markdown_it import MarkdownIt
        replacements=self.tables(s)
        replacements['SHOWCASE_TABLE']=table(['编号','物体名称（源数据）','UID','注册图','Val FG-PSNR','Test FG-PSNR','匹配图','3DGS GIF'],[(r['index'],r['name'],r['uid'],f"{r['registered']}/6",r.get('val_foreground_psnr'),r.get('test_foreground_psnr'),f"[查看]({r['match']})",f"[播放]({r['gif']})") for r in showcase])
        replacements['SHOWCASE_CARDS']='\n\n'.join(f'''### 展示 {r['index']}：{r['name']}

UID：`{r['uid']}`。注册 {r['registered']}/6 张；测试前景 PSNR {fmt(r.get('test_foreground_psnr'))} dB。模型训练 3000 步；GIF 累计时长 3 秒，保留72张原始渲染PNG的哈希证据。

![实际网络输入上的匹配连线]({r['match']})

![居中环绕3DGS渲染]({r['gif']})

[实际 fine_1280 网络输入]({r['source_view']}) · [前景掩码]({r['mask']}) · [源模型许可：{r['license']}]({r['source_url']})
''' for r in showcase)
        # Keep a single canonical Markdown report. Only media URL prefixes differ
        # between the repository and the optional offline export.
        text=(REPO/'docs/assignment2_report.md').read_text().replace('](../assets/','](assets/')
        for k,v in replacements.items():text=text.replace('{{'+k+'}}',v)
        assert '{{' not in text,'Unresolved report placeholder'
        (self.output/'report.md').write_text(text)
        # Exporting evidence must not overwrite the canonical report's prose.
        body=MarkdownIt('commonmark',{'html':True}).enable('table').render(text)
        page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Assignment 2 — 3D Reconstruction and Rendering</title><style>
body{font:16px/1.8 system-ui,"Noto Sans CJK SC",sans-serif;color:#243446;background:#edf2f6;margin:0}main{max-width:1100px;margin:30px auto;padding:35px;background:white}h1,h2,h3{color:#163f52}h2{border-bottom:2px solid #acd8ce;padding-bottom:8px;margin-top:45px}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto}td,th{padding:8px;border:1px solid #dbe5e9;vertical-align:top}th{background:#eaf5f2}img{max-width:100%;height:auto;display:block;margin:15px auto}img[alt*="3DGS"]{max-width:420px}code{overflow-wrap:anywhere;background:#f2f5f7}pre{white-space:pre-wrap;padding:18px;background:#f2f5f7}a{color:#126e87}blockquote{border-left:4px solid #dda44c;padding:8px 15px;background:#fff8eb;margin:15px 0}@media print{body{background:white;font-size:11pt}main{margin:0;padding:0;max-width:none}h2,h3{break-after:avoid}img{max-height:350px}a{color:inherit}table{font-size:9pt}tr{break-inside:avoid}}
</style></head><body><main><p>离线报告：解压整个目录后打开本页；浏览器可打印为PDF，动画以GIF附件为准。</p>'''+body+'</main></body></html>'
        (self.output/'report.html').write_text(page)
        overview='''# ReconLab — 3D Reconstruction and Rendering

从多视角RGB恢复相机和稀疏几何，再优化3D Gaussian Splatting并渲染新视角。

**主流程：** 6张图 → RoMa v2 → 前景与overlap≥0.5过滤 → RANSAC / SfM → 3000步3DGS → 留出视角评估与居中环绕GIF。

[Pipeline documentation](docs/assignment2_report.md) · [实现与复现](experiments/roma100/README.md) · [提示词记录](prompts/user_requests.json) · [实验数据表](assets/tables) · [通用接口](src/reconlab/interfaces.py)

新增的[6／12／24／36视角对照](experiments/view_count/README.md)已完成，简要分析已补入报告第3.5节；对应代码与逐阶段指标已归档。下方10例GIF保留原6图、3k步展示结果。

组号、组员及贡献见报告中的待填写栏。报告按作业1清洗后的Objaverse数据作为输入来源的前提编写。GitHub仓库：[https://github.com/ciErraRuis/3dv_assignment2](https://github.com/ciErraRuis/3dv_assignment2)。

## Doll 多视角推理示例

仓库自带[6张Doll参考图及运行说明](assets/doll/README.md)。[推理入口](scripts/infer.py)默认使用这组asset，执行RoMa匹配、前景与分数过滤、SfM、3000步高斯优化及居中GIF生成；输入不含GT相机。正式报告第2.5节也提供了完整命令。

## 数据与评价

每物体源数据48张512×512图像；本轮固定6张训练、6张验证、6张测试。匹配器内部完整画面等比缩放至800/1280，保存实际输入和哈希。主路线估计相机参数，GT仅用于训练后的对齐与评估；最终方法说明和分析见报告，实验数据表仅保留正式报告引用的过滤与视角数量对照。GIF为已训练高斯渲染的72张512×512帧、24fps、总长3秒，编码可能合并相同帧。

以下10例由用户指定用于展示，不改变原10例对照或固定100例的统计分母。这里的分数为各物体6张留出图的平均值；展示集不用于估计总体成功率。

## 十物体匹配与渲染

'''+replacements['SHOWCASE_TABLE']+'\n\n'+replacements['SHOWCASE_CARDS']+'''
## 提交目录

- `assets/doll`：命令示例使用的6张多视角输入及哈希、来源许可。
- `assets/showcase`、`assets/figures`、`assets/tables`：10例全部匹配结果、连线图、居中GIF、[10例总览GIF](assets/showcase_overview.gif)、输入溯源及正式分析所需统计表。
- `configs/roma100.json`：最终RoMa＋过滤＋SfM＋3DGS的实际参数与来源清单。
- `configs/view_count.json`：报告中的视角数量对照参数。
- `experiments/roma100`、`experiments/view_count`：主实验及视角对照实现，共用`src`中的算法模块。

## 原始证据与代码

各物体目录包含全部15对匹配的NPZ、4对代表性连线图、前景mask、过滤统计、SfM相机和点云、指标、训练记录及GIF。原始输入RGB、实际网络输入示例及完整预处理哈希随结果保留；完整未过滤对应与其余网络输入PNG保存在原实验存储，不重复提交。`matches/*.npz`保存坐标、分数和几何内点；`geometry.npz`保存恢复几何。72张原始环绕PNG的哈希另存，完整PNG及高斯checkpoint按溯源记录保存在实验存储，不提交大权重。

- [当前实验源码快照与环境说明](experiments/roma100/README.md)
- [报告原始统计表](assets/tables)、[来源与哈希](assets/artifact_manifest.json)
- [源物体清单与许可](configs/roma100.json)：每个展示卡片亦链接到原模型来源。

提示词文件收录目前可追溯的任务相关用户请求；不是所有历史模型调用的完整导出，也不包含系统指令或凭据。正式提交前仍需核对提示词覆盖范围。
'''
        for row in showcase:
            needle=f"UID：`{row['uid']}`。"
            overview=overview.replace(needle,needle+f" [全部匹配文件](assets/showcase/{row['uid']}/matches) · [全部连线图](assets/showcase/{row['uid']}/visuals)。")
        (REPO/'README.md').write_text(overview)
        gallery='# 十物体匹配与高斯渲染\n\n[Pipeline documentation](report.html)\n\n展示集与原10例对照集分开；GIF均来自前景＋overlap≥0.5的实际3DGS结果。\n\n'+replacements['SHOWCASE_TABLE']+'\n\n'+replacements['SHOWCASE_CARDS']
        gallery_body=MarkdownIt('commonmark',{'html':True}).enable('table').render(gallery)
        (self.output/'showcase.html').write_text(page[:page.index('<main>')]+'<main>'+gallery_body+'</main></body></html>')
        self.export_csv('showcase_10',showcase)
    def run(self):
        collector=EvidenceCollector();s=collector.collect()
        assert len(s['showcase_uids'])==len(set(s['showcase_uids']))==10
        publisher=ArtifactPublisher(self.output);showcase=publisher.showcase(s)
        self.plots(s)
        for k in ['frontend','ablation','gt','views_historical','expansion']:self.export_csv(k,s[k])
        json_write(self.output/'evidence/experiment_snapshot.json',s)
        json_write(self.output/'evidence/showcase_selection.json',collector.selection)
        for p in [RUNTIME/'config.json',RUNTIME/'MANIFEST.json',REPO/'prompts/user_requests.json',REPO/'experiments/roma100/provenance/runtime_versions.json',REPO/'experiments/roma100/provenance/upstream_sources.json']:
            publisher.copy(p,Path('evidence')/p.name)
        self.render(s,publisher,showcase)
        json_write(self.output/'artifact_manifest.json',{'generated_at':s['snapshot_finished_utc'],'copied_files':publisher.files,'showcase':showcase,'scope':'10 user-selected objects; benchmark cohort unchanged; PNG sequences/checkpoints retained at recorded execution paths'})
        for p in (self.output/'assets').rglob('*'):
            if p.is_file() and submission_asset(p.relative_to(self.output/'assets')):
                target=REPO/'assets'/p.relative_to(self.output/'assets')
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        for name in ['ablation','showcase_10']:
            shutil.copyfile(self.output/'tables'/f'{name}.csv',REPO/'assets/tables'/f'{name}.csv')
        manifest=read(self.output/'artifact_manifest.json')
        manifest['copied_files']={k:v for k,v in manifest['copied_files'].items()
                                  if not k.startswith('assets/') or submission_asset(k[7:])}
        manifest['github_export']={'scope':'Complete filtered matches; original RGB; one actual fine input per object; full preprocessing hashes and audit counts; centered GIFs. Raw audits and other network PNGs remain in original experiment storage.'}
        json_write(REPO/'assets/artifact_manifest.json',manifest)
        print(json.dumps({'output':str(self.output),'showcase':len(showcase),'copied_files':len(publisher.files),'snapshot':s['snapshot_finished_utc'],'expansion_counts':dict(Counter(r['state'] for r in s['expansion']))},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DATA/'roma100_filter_r01/assignment2_report');args=parser.parse_args();AssignmentReportBuilder(args.output).run()
