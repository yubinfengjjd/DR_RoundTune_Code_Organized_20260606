from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path

OUT = Path("chatgpt_exports/IBD_Mucosal_Spatial_Niche_Review_Outline_CN.docx")
OUT.parent.mkdir(parents=True, exist_ok=True)

doc = Document()
sec = doc.sections[0]
sec.top_margin = Cm(2.0)
sec.bottom_margin = Cm(2.0)
sec.left_margin = Cm(2.2)
sec.right_margin = Cm(2.0)

# -------- styles --------
for s in ["Normal", "Title", "Heading 1", "Heading 2", "Heading 3"]:
    st = doc.styles[s]
    st.font.name = "Times New Roman"
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
doc.styles["Normal"].font.size = Pt(10.5)
doc.styles["Title"].font.size = Pt(18)
doc.styles["Heading 1"].font.size = Pt(15)
doc.styles["Heading 2"].font.size = Pt(12.5)
doc.styles["Heading 3"].font.size = Pt(11)

def font(run, size=10.5, bold=False, color=None, east="宋体"):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east)
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)

def p(text="", bold=False, center=False, size=10.5, color=None, first_indent=True, after=4):
    par = doc.add_paragraph()
    par.paragraph_format.space_after = Pt(after)
    if first_indent:
        par.paragraph_format.first_line_indent = Cm(0.74)
    if center:
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = par.add_run(text)
    font(r, size, bold, color)
    return par

def h(text, level=1):
    par = doc.add_paragraph()
    par.paragraph_format.space_before = Pt(8 if level == 1 else 5)
    par.paragraph_format.space_after = Pt(4)
    r = par.add_run(text)
    if level == 1:
        font(r, 15, True, "17365D", "黑体")
    elif level == 2:
        font(r, 12.5, True, "245B78", "黑体")
    else:
        font(r, 11.2, True, "3B6A57", "黑体")
    return par

def bullets(items, indent=0):
    for x in items:
        par = doc.add_paragraph()
        par.paragraph_format.left_indent = Cm(0.55 + indent*0.55)
        par.paragraph_format.first_line_indent = Cm(-0.28)
        par.paragraph_format.space_after = Pt(2)
        r = par.add_run("• " + x)
        font(r, 10.2)

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)

def table(headers, rows, font_size=8.8):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, hd in enumerate(headers):
        c = t.rows[0].cells[i]
        shade(c, "DCEAF5")
        c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        q = c.paragraphs[0]
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = q.add_run(hd)
        font(r, 9.2, True, "17365D", "黑体")
    for row in rows:
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            rr = cells[j].paragraphs[0].add_run(str(val))
            font(rr, font_size)
    doc.add_paragraph()
    return t

# -------- cover --------
p("肠黏膜“空间生态位”与炎症性肠病（IBD）", True, True, 20, "17365D", False, 2)
p("上皮–成纤维细胞–免疫细胞串扰：高影响力综述思路与提纲", True, True, 13.5, "245B78", False, 2)
p("定位：IF >10；面向 Nature Reviews / Cell Trends / CNS 子刊风格机制综述", False, True, 10.5, "5B6D7A", False, 8)
p("建议英文题目：The mucosal spatial niche in inflammatory bowel disease: epithelial–fibroblast–immune crosstalk from injury to failed repair", False, True, 10.5, None, False, 8)

doc.add_page_break()

h("1. 这篇综述真正应该写什么：核心定位", 1)
p("本综述不建议写成“上皮细胞一节、成纤维细胞一节、免疫细胞一节”的平行综述，也不建议简单总结单细胞研究发现。更有竞争力的组织方式，是把IBD黏膜看成一个具有明确空间层级、动态状态转换和跨细胞反馈的组织生态系统。文章的中心问题不是“有哪些细胞”，而是“这些细胞在什么位置、以什么状态、通过什么信号网络共同决定黏膜是恢复稳态、持续炎症，还是进入纤维化和难治状态”。")

p("建议全文唯一主线：", True)
p("Homeostatic niche → epithelial injury/barrier breach → inflammatory recruitment → stromal activation → failed resolution → repair versus fibrosis → mechanism-based patient stratification", True, True, 11.5, "8A3150", False, 8)

h("2. 综述的差异化卖点", 1)
bullets([
    "从“细胞组成”升级到“空间生态位”：强调lumen–mucus–epithelium–lamina propria–vascular/perivascular compartment的空间连续体。",
    "从“细胞类型”升级到“细胞状态”：尤其关注inflammatory fibroblast、myofibroblast、crypt-supportive stromal cells、repair epithelial states和inflammatory myeloid states。",
    "从“炎症机制”升级到“状态转换”：把稳态、急性损伤、慢性炎症、修复、纤维化视为可转换的黏膜niche states。",
    "从“单细胞关联”升级到“空间因果”：强调ligand–receptor分析只能生成假设，必须结合空间共定位、organoid/co-culture和原位验证。",
    "从“疾病名称分层”升级到“mechanistic endotypes”：提出epithelial-injury、myeloid-inflammatory、stromal-activated、fibrosis-prone、vascular-remodeling、repair/healing等生态位内表型。",
    "从“抑炎治疗”升级到“组织稳态重建”：终点不仅是降低炎症，还包括mucosal healing、barrier restoration、stromal normalization和fibrosis prevention。"
])

h("3. 五个核心科学问题", 1)
bullets([
    "空间问题：IBD黏膜中不同细胞群位于哪里？它们与crypt、surface epithelium、blood vessel及ECM之间具有怎样的空间关系？",
    "状态问题：稳态成纤维细胞如何转变为inflammatory fibroblast或myofibroblast？上皮细胞如何从损伤状态进入再生或耗竭？",
    "串扰问题：哪些epithelial–fibroblast–immune ligand–receptor轴真正具有驱动性，而不是转录组相关？",
    "时间问题：急性炎症为何在部分患者中顺利消退，而在另一些患者中进入慢性活化、纤维化和难治状态？",
    "转化问题：能否用空间生态位特征预测药物反应、复发、纤维化和手术风险，并据此选择治疗？"
])

h("4. 推荐正文结构", 1)

outline = [
("4.1 Introduction — IBD is a disease of tissue organization, not merely immune activation", [
"概述传统IBD框架：屏障缺陷、菌群失衡、遗传易感与异常免疫。",
"指出传统框架的缺口：相同炎症通路在不同患者中产生不同结局，提示组织空间背景决定信号效应。",
"提出“mucosal spatial niche”概念：细胞状态、位置、邻域关系、ECM、血管、微生物与代谢物共同组成动态生态系统。",
"明确本文重点是上皮–成纤维细胞–免疫细胞三角网络，并将血管、神经和菌群作为调控边界。"
]),
("4.2 Architecture of the healthy mucosal niche", [
"从肠腔和黏液层开始，建立空间层级：microbiota → mucus → surface/crypt epithelium → lamina propria → submucosal vasculature。",
"上皮细胞：absorptive cells、goblet cells、stem/progenitor cells、secretory states。",
"稳态间质：crypt-base stromal cells、surface-like fibroblasts、perivascular stromal cells。",
"免疫稳态：resident macrophages、Tregs、ILCs、dendritic cells和IgA-producing plasma cells。",
"强调Wnt/R-spondin/BMP modulators、EGF、TGF-β、SCFAs等共同维持空间稳态。"
]),
("4.3 Epithelial injury as the entry point of niche collapse", [
"黏液减少、tight-junction破坏、上皮死亡与通透性升高。",
"微生物PAMPs及宿主DAMPs进入lamina propria，触发myeloid sensing。",
"上皮不只是被动屏障：损伤上皮释放alarmins、cytokines和growth factors，主动重塑免疫与间质。",
"重点讨论stem/progenitor-cell exhaustion、aberrant regeneration及损伤相关上皮状态。"
]),
("4.4 Fibroblast heterogeneity and spatial specialization", [
"这是全文核心章节之一，不把fibroblast视为单一细胞群。",
"Crypt-base trophocyte / stem-cell-supportive fibroblast：Wnt、R-spondins、BMP antagonists。",
"Surface/villus-like fibroblasts：支持成熟上皮和屏障稳态。",
"Inflammatory fibroblasts：IL-6、IL-11、OSM、CCL2、CXCL8、CXCL12及MMPs。",
"Myofibroblasts：TGF-β驱动收缩、ECM沉积与组织重塑。",
"Perivascular fibroblasts/pericyte-like stromal cells：连接血管、免疫细胞迁移与angiogenesis。",
"Fibrosis-associated fibroblasts：持续ECM沉积、LOX介导交联、组织僵硬和stricture形成。"
]),
("4.5 Immune cells as niche remodelers rather than isolated effectors", [
"Macrophage/monocyte：炎症放大、efferocytosis与修复之间的状态转换。",
"Neutrophil：急性杀菌和组织损伤的双重作用。",
"DC–T-cell轴：IL-23–Th17/Th1与Treg平衡。",
"ILCs：将上皮alarmins、微生物信号和修复程序连接起来。",
"B/plasma cells：抗体、免疫复合物与慢性黏膜生态。",
"强调免疫细胞通过cytokines/chemokines反过来改变fibroblast和epithelial states。"
]),
("4.6 Epithelial–fibroblast–immune feedback circuits in active IBD", [
"TNF–TNFR、IL-1β–IL-1R、IL-6–gp130、IL-11–IL-11R、OSM–OSMR、IL-23–IL-23R、IL-17–IL-17R等炎症轴。",
"Wnt/R-spondin/BMP/EGF等再生轴。",
"CCL2–CCR2、CXCL12–CXCR4、CXCL8相关髓系招募轴。",
"integrin–MAdCAM/ICAM/VCAM介导的空间迁移和滞留。",
"构建“barrier breach → immune recruitment → fibroblast activation → ECM/cytokine remodeling → persistent epithelial dysfunction”的自我放大环。"
]),
("4.7 From inflammation to repair or fibrosis: the niche-state transition", [
"把全文从静态机制提升为时间序列。",
"Stage 1：homeostatic niche。",
"Stage 2：epithelial-injury / flare niche。",
"Stage 3：chronic inflammatory niche。",
"Stage 4A：repair/healing niche。",
"Stage 4B：stromal-activated / fibrosis-prone niche。",
"核心问题：决定分叉点的是哪些resolution checkpoints？包括Treg/IL-10、pro-resolving macrophages、MMP/TIMP balance、fibroblast deactivation和barrier restoration。"
]),
("4.8 UC versus Crohn’s disease: shared architecture, different spatial outputs", [
"不要把UC和CD完全分开写，而是比较相同niche模块在两类疾病中的不同输出。",
"UC：黏膜浅层炎症、上皮/黏液与表浅间质生态位更突出。",
"CD：深层组织、transmural inflammation、perivascular/perienteric stroma、fibrosis/stricture和fistula更突出。",
"将疾病差异转化为空间深度、细胞状态和修复结局的差异。"
]),
("4.9 Technology section — how to actually resolve the mucosal niche", [
"scRNA-seq：解析状态与稀有细胞，但丢失空间信息。",
"Spatial transcriptomics：建立细胞状态与组织结构对应关系。",
"Multiplex imaging：验证细胞邻域和蛋白表达。",
"Organoid–fibroblast–immune co-culture：做真正的功能扰动和因果验证。",
"Lineage tracing/fate mapping：回答细胞来源、可塑性和状态转换。",
"Microbiome/metabolome integration：把腔内环境接到组织生态。",
"Cell–cell communication algorithms：仅作为hypothesis-generating工具。",
"In situ validation：RNAscope、IHC/IF、PLA等完成患者组织验证。"
]),
("4.10 Translational targeting of the mucosal niche", [
"从单靶点免疫抑制转向分层修复：immune control + epithelial repair + stromal normalization + vascular normalization。",
"现有抗TNF、IL-23、JAK、integrin等治疗可以重新解释为对不同niche模块的干预。",
"未来方向：anti-OSM/OSMR、anti-IL-11、抗纤维化、ECM重塑、上皮再生、细胞治疗和靶向递送。",
"提出机制分层：不同患者可能需要epithelial-first、immune-first或stroma-first策略。"
]),
("4.11 Future perspective — from mucosal healing to niche normalization", [
"终点不应只停留在endoscopic healing。",
"未来更高级的目标是恢复细胞比例、空间邻域、ECM结构、血管稳态和屏障功能。",
"建立可重复的spatial niche score与multi-modal biomarker panel。",
"最终形成mechanism-based enrollment和niche-directed therapy。"
])
]

for title, bullets_ in outline:
    h(title, 2)
    bullets(bullets_)

doc.add_page_break()
h("5. 建议的6幅主图", 1)
fig_rows = [
["Figure 1","The mucosal spatial niche in IBD","全文总览：lumen–mucus–epithelium–lamina propria–vascular niche；比较稳态、活动性炎症和修复/纤维化。","Introduction后"],
["Figure 2","Epithelial–fibroblast crosstalk across injury, repair and fibrosis","三联图：homeostasis → acute injury → repair/fibrosis；突出Wnt/R-spondin/BMP、EGF、TGF-β、MMP/ECM。","上皮与fibroblast章节之间"],
["Figure 3","Spatial heterogeneity of fibroblast niches in IBD","把crypt-base、surface-like、inflammatory、myofibroblast、perivascular和fibrosis-associated fibroblasts放到实际空间位置。","Fibroblast核心章节后"],
["Figure 4","Epithelial–immune–fibroblast feedback circuits in active colitis","barrier breach → PAMP/DAMP → myeloid/T cell recruitment → fibroblast activation → failed resolution。","串扰章节后"],
["Figure 5","Evolution of the mucosal niche from homeostasis to chronic remodeling","时间轴/状态转换图：homeostasis → flare → chronic active → healing或fibrosis。","疾病进展章节后"],
["Figure 6","Therapeutic and translational targeting of the mucosal spatial niche","按microbiota、epithelium、immune、fibroblast、vascular和biomarker六层治疗映射。","治疗与Perspective章节"]
]
table(["图号","建议标题","核心信息","插入位置"], fig_rows, 8.6)

h("6. 建议的4个主表", 1)
tab_rows = [
["Table 1","Major cellular components of the mucosal spatial niche in IBD","Cell type；principal location；homeostatic function；pathogenic role；markers/signals","建立全文细胞词典"],
["Table 2","Key epithelial–fibroblast–immune signaling axes","Ligand–receptor；source；target；effect；therapeutic relevance","把机制从细胞描述升级到可干预信号轴"],
["Table 3","Spatial niche states and disease endotypes in IBD","Niche state；spatial features；dominant cells/pathways；clinical correlate；intervention","全篇最具转化价值的表"],
["Table 4","Emerging technologies for studying the mucosal niche","Technology；what it reveals；readouts；strengths；limitations","体现方法学成熟度与未来方向"]
]
table(["表号","建议标题","建议栏目","用途"], tab_rows, 8.6)

h("7. 建议在正文反复使用的6个概念", 1)
bullets([
    "Spatial niche：空间生态位，而非单纯cell type。",
    "Cell state：疾病相关状态比传统细胞分类更重要。",
    "Neighborhood：某细胞周围是谁，往往比该细胞单独表达什么更具解释力。",
    "Niche transition：从稳态到炎症、修复或纤维化的动态转换。",
    "Failed resolution：慢性IBD不是“炎症一直开着”这么简单，而是终止炎症和组织恢复程序失败。",
    "Niche normalization：比mucosal healing更高阶的未来治疗终点。"
])

h("8. 审稿人最容易质疑的地方", 1)
bullets([
    "不要把单细胞通讯算法预测的ligand–receptor关系直接写成已证实的细胞串扰；必须标注为predicted或candidate interaction。",
    "不要把所有PDGFRα+或COL1A1+细胞笼统称为fibroblast；不同研究的注释体系需要建立跨队列对应关系。",
    "不要把fibroblast activation一概等同于纤维化；急性修复中短暂激活可能是必要的。",
    "不要把UC和CD完全混合；应明确空间深度、transmural involvement和纤维化倾向的差异。",
    "不要把mucosal healing等同于真正恢复稳态；临床和内镜缓解后仍可能存在分子/空间异常。",
    "治疗章节需区分已获批机制、临床开发中机制和纯前临床假说，避免把探索性靶点写成临床可用治疗。"
])

h("9. 写作时建议的篇幅分配", 1)
table(["部分","建议占全文比例","写作重点"],[
["Introduction + conceptual framework","10%","快速提出spatial niche与failed resolution，不做过长背景。"],
["Healthy niche + epithelial injury","15%","建立正常空间结构，再解释最初破坏。"],
["Fibroblast heterogeneity","20%","全文重点，体现差异化。"],
["Immune–stromal–epithelial circuits","20%","避免细胞清单，围绕feedback circuits组织。"],
["Niche transition / fibrosis / UC-CD comparison","15%","把静态机制变成疾病演进。"],
["Technology + translation","15%","空间组学方法和精准治疗。"],
["Perspective / conclusion","5%","提出niche normalization与未来试验框架。"]
], 8.8)

h("10. 题目建议", 1)
bullets([
    "首选：The mucosal spatial niche in inflammatory bowel disease: epithelial–fibroblast–immune crosstalk from injury to failed repair",
    "更偏机制：Spatial organization of epithelial–stromal–immune circuits in inflammatory bowel disease",
    "更偏转化：From mucosal inflammation to niche normalization: spatial ecosystems and therapeutic vulnerabilities in inflammatory bowel disease",
    "中文：肠黏膜空间生态位与炎症性肠病：上皮–成纤维细胞–免疫细胞串扰、组织重塑与精准治疗"
])

h("11. 一句话总论点", 1)
p("IBD不应只被理解为异常免疫反应，而应被理解为一个空间组织失衡的黏膜生态系统疾病：上皮损伤改变免疫感知，免疫炎症重编程成纤维细胞与血管生态，间质和ECM进一步限制上皮再生；疾病是否进入缓解、慢性活动或纤维化，取决于这些空间生态位能否完成从炎症到组织稳态的重置。", True, False, 11.2, "17365D")

h("12. 实际写作顺序建议", 1)
bullets([
    "第一步：先完成Figure 1和Figure 5的逻辑，因为这两幅图决定全文叙事。",
    "第二步：建立跨文献fibroblast taxonomy表，把不同研究对stromal subsets的命名对齐。",
    "第三步：以Table 2为核心整理ligand–receptor证据，分别标记human tissue、mouse perturbation、organoid/co-culture和computational inference。",
    "第四步：再写正文，不要先按传统小标题机械堆文献。",
    "第五步：最后用Table 3把mechanistic niche states映射到临床表型和治疗反应，形成综述的转化卖点。"
])

doc.core_properties.title = "肠黏膜空间生态位与IBD：上皮–成纤维细胞–免疫细胞串扰综述思路提纲"
doc.core_properties.subject = "High-impact review outline"
doc.core_properties.author = "ChatGPT"
doc.save(OUT)

# Basic validation
check = Document(OUT)
assert len(check.paragraphs) > 80
assert len(check.tables) >= 4
assert OUT.stat().st_size > 30000
print(str(OUT))
print("paragraphs=", len(check.paragraphs), "tables=", len(check.tables), "size=", OUT.stat().st_size)
