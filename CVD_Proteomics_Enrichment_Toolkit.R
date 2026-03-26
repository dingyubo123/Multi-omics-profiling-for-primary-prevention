# ==============================================================================
# 1. 环境准备：自动检测并安装缺失的包
# ==============================================================================
packages <- c("BiocManager", "tidyverse", "ggraph")
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")

# Bioconductor 专用包列表
bio_pkgs <- c("clusterProfiler", "ReactomePA", "org.Hs.eg.db", "enrichplot")

# 检查并安装
for (p in bio_pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    message(paste("正在安装包:", p))
    BiocManager::install(p, update = FALSE, ask = FALSE)
  }
}

# 加载包
library(tidyverse)
library(clusterProfiler)
library(ReactomePA)
library(org.Hs.eg.db)
library(enrichplot)

# ==============================================================================
# 2. 读取数据
# ==============================================================================
# 注意：R语言中 Windows 路径需要用双反斜杠 "\\" 或 单斜杠 "/"
file_path <- "C:\\Users\\Administrator\\Desktop\\stable_features1.csv"

message("正在读取文件...")
if (file.exists(file_path)) {
  df <- read.csv(file_path)
  message("文件读取成功！")
} else {
  stop("错误：找不到文件，请检查路径是否正确！")
}

# 检查是否存在 Protein 列
if (!"Protein" %in% colnames(df)) {
  stop("错误：CSV文件中找不到 'Protein' 列，请检查列名拼写。")
}

# 提取基因列表
gene_list <- as.character(df$Protein)
message(paste("共提取到", length(gene_list), "个基因。"))

# ... (在你的代码第2步读取完 df 之后，第3步 bitr 之前插入) ...

# 1. 提取基因列表 (假设你之前的变量名是 gene_list)
gene_list <- as.character(df$Protein)

message("正在修正不规范的基因名...")

# 修正 1: 处理 NTproBNP
# 因为它是 NPPB 的产物，我们把它改为 NPPB。
# 即使列表里已经有了 NPPB 也没关系，我们后面会去重。
gene_list[gene_list == "NTproBNP"] <- "NPPB"

# 修正 2: 处理 MICB_MICA
# 找到 MICB_MICA 的位置
idx <- which(gene_list == "MICB_MICA")
if (length(idx) > 0) {
  # 移除 MICB_MICA
  gene_list <- gene_list[-idx]
  # 追加分开的两个基因
  gene_list <- c(gene_list, "MICA", "MICB")
}

# 3. 去重 (非常重要！因为 NTproBNP 改名后可能和原有的 NPPB 重复)
gene_list <- unique(gene_list)

message(paste("修正并去重后，剩余基因数量:", length(gene_list)))

# ... (然后继续运行你原来的第3步 ids <- bitr(...)) ...

# ==============================================================================
# 3. ID 转换 (Symbol -> Entrez ID)
# ==============================================================================
message("正在进行基因 ID 转换...")

# Reactome 分析需要 Entrez ID，这里利用 org.Hs.eg.db 进行转换
ids <- bitr(gene_list, 
            fromType = "SYMBOL", 
            toType = "ENTREZID", 
            OrgDb = "org.Hs.eg.db")

message(paste("成功转换", nrow(ids), "个基因 ID。"))
# 提示：如果有基因转换失败，bitr 会自动忽略并发出警告，这是正常的

# ==============================================================================
# 4. 运行 Reactome 通路富集分析
# ==============================================================================
message("正在运行 Reactome 通路富集分析，请稍候...")

# 运行富集分析
# pvalueCutoff = 0.05 是常用标准，如果结果太少，可尝试调整为 0.1
x <- enrichPathway(gene = ids$ENTREZID, 
                   pvalueCutoff = 0.05,
                   readable = TRUE) # readable=TRUE 会让图上显示基因名而不是数字ID

# ==============================================================================
# 5. 绘图与保存
# ==============================================================================

if (is.null(x) || nrow(x@result) == 0) {
  warning("未找到显著富集的通路！可能是基因数量太少，或这些基因没有显著聚集在某些通路中。")
} else {
  message("分析完成，正在绘图...")
  
  # 绘制 Cnetplot (基因-通路网络图)
  # circular = FALSE 表示使用网络布局（类似你提供的图 C）
  # showCategory = 5 表示只展示最显著的前5条通路，避免图太乱，可自行修改数字
  p <- cnetplot(x, 
                circular = FALSE, 
                colorEdge = TRUE, 
                showCategory = 5, 
                cex_label_category = 1.2, # 通路字体大小
                cex_label_gene = 0.8)     # 基因字体大小
  
  print(p)
  
  # 保存图片到桌面
  output_file <- "C:\\Users\\Administrator\\Desktop\\Reactome_Network.pdf"
  ggsave(output_file, plot = p, width = 10, height = 8)
  
  message(paste("图片已保存至:", output_file))
}
# ==============================================================================
# 修正后的代码：1. 提取“边”文件 (Edges)
# ==============================================================================
library(tidyverse) # 确保 tidyverse 已加载

# 提取富集结果 (确保 x 对象还在环境中)
if (!exists("x")) {
  stop("错误：对象 'x' 不存在，请先运行上面的 Reactome 分析代码！")
}

results_df <- as.data.frame(x)

# 这里的关键修正是：使用了 dplyr::select 而不是直接用 select
edges <- results_df %>%
  dplyr::select(Description, geneID) %>%   # <--- 强制使用 dplyr 包的 select
  separate_rows(geneID, sep = "/") %>%     # 将 "GeneA/GeneB" 拆行
  dplyr::rename(Source = Description, Target = geneID) %>% # <--- 强制使用 dplyr 的 rename
  mutate(Interaction = "Pathway-Gene")

# 保存边文件
write.csv(edges, "C:\\Users\\Administrator\\Desktop\\cytoscape_edges.csv", row.names = FALSE)
message("成功！边文件已导出: cytoscape_edges.csv")

# ==============================================================================
# 2. 提取“节点”文件 (Nodes) - 定义节点的类型和属性
# ==============================================================================

# A. 通路节点 (Source)
pathway_nodes <- data.frame(
  id = unique(edges$Source),
  type = "Pathway",
  shape_type = "Circle" 
)

# B. 基因节点 (Target)
gene_nodes <- data.frame(
  id = unique(edges$Target),
  type = "Gene",
  shape_type = "Circle" 
)

# C. 标记特殊的基因 (这里你可以根据你的需要修改)
# 比如这里我想标记那些在 stable_features1.csv 原文件里的蛋白
# 假设原来的基因列表叫 gene_list (就是你之前代码里读取的)
# 如果之前的 gene_list 还在，可以直接用；如果不在，这里先全部设为 Circle
if (exists("gene_list")) {
  # 比如我们想突出显示输入列表中的前 5 个基因 (作为演示)
  # 实际应用中，你可能想标记所有输入的基因
  special_genes <- gene_list 
  gene_nodes$shape_type <- ifelse(gene_nodes$id %in% special_genes, "Square", "Circle")
}

# 合并节点表
all_nodes <- rbind(pathway_nodes, gene_nodes)

# 保存节点文件
write.csv(all_nodes, "C:\\Users\\Administrator\\Desktop\\cytoscape_nodes.csv", row.names = FALSE)
message("成功！节点文件已导出: cytoscape_nodes.csv")

# 将富集分析的详细统计结果（包括 P值、基因数、FDR等）保存为 CSV
write.csv(as.data.frame(x), "C:\\Users\\Administrator\\Desktop\\Enrichment_Results_Table.csv")

# ==============================================================================
# 4.5 补充运行 GO 富集分析 (BP, CC, MF)
# ==============================================================================
message("正在运行 GO 富集分析 (BP/CC/MF)...")

# 定义一个函数来简化重复操作
run_go_analysis <- function(ont_type) {
  message(paste("正在分析:", ont_type, "..."))
  
  ego <- enrichGO(gene          = ids$ENTREZID,
                  OrgDb         = org.Hs.eg.db,
                  ont           = ont_type,    # "BP", "CC", "MF" 或 "ALL"
                  pAdjustMethod = "BH",
                  pvalueCutoff  = 0.05,
                  qvalueCutoff  = 0.2,
                  readable      = TRUE)        # 自动将 ID 转回基因名
  return(ego)
}

# 1. 运行 BP (生物过程 - 最常用)
go_bp <- run_go_analysis("BP")

# 2. 运行 CC (细胞组分)
go_cc <- run_go_analysis("CC")

# 3. 运行 MF (分子功能)
go_mf <- run_go_analysis("MF")

# ==============================================================================
# 保存 GO 结果到 CSV
# ==============================================================================
if (!is.null(go_bp)) write.csv(as.data.frame(go_bp), "C:\\Users\\Administrator\\Desktop\\GO_BP_Results.csv")
if (!is.null(go_cc)) write.csv(as.data.frame(go_cc), "C:\\Users\\Administrator\\Desktop\\GO_CC_Results.csv")
if (!is.null(go_mf)) write.csv(as.data.frame(go_mf), "C:\\Users\\Administrator\\Desktop\\GO_MF_Results.csv")

message("GO 分析完成，结果已保存！")

# ==============================================================================
# 5. 绘制 BP, CC, MF 三合一分面气泡图 (最佳展示方式)
# ==============================================================================
library(ggplot2)
library(dplyr) # 确保加载 dplyr 用于数据处理

message("正在绘制三合一气泡图...")

# 1. 检查对象是否存在
if (!exists("go_bp") || !exists("go_cc") || !exists("go_mf")) {
  stop("错误：未找到 go_bp/go_cc/go_mf 对象，请确保上面的分析代码已成功运行！")
}

# 2. 定义一个提取数据的辅助函数
# 作用：提取前 N 个最显著的通路，并打上标签 (BP/CC/MF)
get_top_go_data <- function(go_obj, type_label, top_n = 10) {
  # 如果结果为空，返回 NULL
  if (is.null(go_obj) || nrow(go_obj@result) == 0) return(NULL)
  
  df <- as.data.frame(go_obj) %>%
    arrange(p.adjust) %>%   # 按显著性排序
    head(top_n) %>%         # 取前 top_n 个
    mutate(Type = type_label) # 添加分类标签
  return(df)
}

# 3. 提取数据 (这里我们取前 10 个，你可以修改 top_n = 15 或其他数字)
df_bp <- get_top_go_data(go_bp, "Biological Process (BP)", top_n = 10)
df_cc <- get_top_go_data(go_cc, "Cellular Component (CC)", top_n = 10)
df_mf <- get_top_go_data(go_mf, "Molecular Function (MF)", top_n = 10)

# 4. 合并数据
df_all <- rbind(df_bp, df_cc, df_mf)

# 检查是否有数据
if (is.null(df_all) || nrow(df_all) == 0) {
  warning("没有足够的数据用于绘图！")
} else {
  
  # 5. 数据处理：计算 GeneRatio 为数值 (用于 X 轴)
  # 原始 GeneRatio 是 "10/115" 这样的字符串，无法直接画图，需转为小数
  df_all$GeneRatio_Num <- sapply(df_all$GeneRatio, function(x) {
    nums <- as.numeric(unlist(strsplit(x, "/")))
    return(nums[1] / nums[2])
  })
  
  # 6. 设置显示顺序 (BP 在上，CC 中，MF 在下)
  df_all$Type <- factor(df_all$Type, 
                        levels = c("Biological Process (BP)", 
                                   "Cellular Component (CC)", 
                                   "Molecular Function (MF)"))
  
  # 7. 开始绘图 (ggplot2)
  p <- ggplot(df_all, aes(x = GeneRatio_Num, 
                          y = reorder(Description, GeneRatio_Num))) + # 按比例大小排序Y轴
    
    # 绘制气泡
    geom_point(aes(size = Count, color = p.adjust)) +
    
    # 【核心步骤】分面显示：把 BP, CC, MF 分开画在同一张图上
    facet_grid(Type ~ ., scales = "free_y", space = "free_y") +
    
    # 颜色设置：红色代表显著 (p.adjust数值小)，蓝色代表不显著
    scale_color_gradient(low = "#E41A1C", high = "#377EB8", 
                         name = "p.adjust", 
                         guide = guide_colorbar(reverse = TRUE)) +
    
    # 标题和坐标轴
    labs(title = "GO Enrichment Analysis",
         subtitle = "Top 10 Terms for BP, CC, and MF",
         x = "GeneRatio", 
         y = NULL) +
    
    # 美化图表主题
    theme_bw() +
    theme(strip.text = element_text(size = 15, face = "bold", color = "black"), # 分面标题加粗
          strip.background = element_rect(fill = "#EFEFEF"), # 分面背景灰底
          axis.text.y = element_text(size = 10, color = "black"), # Y轴文字清晰
          axis.title.x = element_text(size = 15)) 
  
  # 8. 显示并保存
  print(p)
  
  # 保存 PDF (矢量图，适合投稿)
  ggsave("C:\\Users\\Administrator\\Desktop\\GO_All_Combined_Dotplot.pdf", p, width = 10, height = 15)
  message("图片已保存：GO_All_Combined_Dotplot.pdf")
}

# ==============================================================================
# 5. 补充进阶分析：KEGG 和 DO (疾病)
# ==============================================================================

# --------------------------------------------------------------------------
# A. 运行 KEGG 通路分析
# --------------------------------------------------------------------------
# 注意：KEGG 需要联网，偶尔会因为网络问题报错，多试几次即可
message("正在运行 KEGG 分析...")

kk <- enrichKEGG(gene         = ids$ENTREZID,
                 organism     = 'hsa',   # hsa 代表人类 (homo sapiens)
                 pvalueCutoff = 0.05)

if (!is.null(kk)) {
  # 将结果稍微处理一下（把 ID 变回基因名，方便阅读）
  kk <- setReadable(kk, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
  
  # 保存
  write.csv(as.data.frame(kk), "C:\\Users\\Administrator\\Desktop\\KEGG_Results.csv")
  message("KEGG 分析完成！")
}

# ==============================================================================
# B. 运行疾病富集分析 (修复版 + 推荐版)
# ==============================================================================
library(DOSE)

# --------------------------------------------------------------------------
# 方案 1: 尝试修复 enrichDO (如果安装了 DO.db 通常就能跑通)
# --------------------------------------------------------------------------
message("正在尝试运行 DO 分析...")

tryCatch({
  do_res <- enrichDO(gene          = ids$ENTREZID,
                     # ont           = "DO",      # <--- 注释掉这一行，使用默认值
                     pvalueCutoff  = 0.05,
                     pAdjustMethod = "BH",
                     readable      = TRUE)
  
  if (!is.null(do_res)) {
    write.csv(as.data.frame(do_res), "C:\\Users\\Administrator\\Desktop\\DO_Results.csv")
    message("DO 分析成功！")
  }
}, error = function(e) {
  message("DO 分析报错，可能是网络或包版本问题，正在跳过...")
})

# --------------------------------------------------------------------------
# 方案 2: 强烈推荐使用 DisGeNET (enrichDGN)
# ==============================================================================
# 推荐：运行 DisGeNET (DGN) 疾病富集分析
# ==============================================================================
library(DOSE)

message("正在运行 DisGeNET (DGN) 疾病富集分析...")

# pvalueCutoff 可以设为 0.05
# qvalueCutoff 可以设为 0.2 或 0.05
dgn_res <- enrichDGN(gene          = ids$ENTREZID,
                     pvalueCutoff  = 0.05,
                     pAdjustMethod = "BH",
                     qvalueCutoff  = 0.2,
                     readable      = TRUE) # 结果自动显示基因名

if (is.null(dgn_res) || nrow(dgn_res) == 0) {
  message("未找到显著富集的 DisGeNET 结果。")
} else {
  # 1. 保存表格
  write.csv(as.data.frame(dgn_res), "C:\\Users\\Administrator\\Desktop\\DisGeNET_Results.csv")
  message("DisGeNET 分析完成！结果已保存。")
  
  # 2. 绘制气泡图 (Dotplot) - 非常直观
  p_dgn <- dotplot(dgn_res, showCategory=15) + 
    ggtitle("DisGeNET Disease Enrichment")
  
  print(p_dgn)
  ggsave("C:\\Users\\Administrator\\Desktop\\DisGeNET_Dotplot.pdf", p_dgn, width = 8, height = 10)
  
  # 3. 绘制网络图 (如果需要类似 Figure C 的效果)
  # 你也可以导出 edges 和 nodes 去 Cytoscape 画
  p_net <- cnetplot(dgn_res, circular = FALSE, colorEdge = TRUE, showCategory = 5)
  ggsave("C:\\Users\\Administrator\\Desktop\\DisGeNET_Network.pdf", p_net, width = 10, height = 8)
}
# ==============================================================================
# ==============================================================================
# 6. 定制化绘图：筛选心血管、炎症、免疫、代谢相关结果
# ==============================================================================
library(tidyverse)

# 1. 确保你有富集分析的结果对象 (dgn_res)
if (!exists("dgn_res")) stop("请先运行上面的 DisGeNET 分析代码生成 dgn_res 对象！")

# 2. 将结果转换为普通的数据框
df_all <- as.data.frame(dgn_res)

# 3. 定义扩展后的关键词列表
# 新增了：Immune (免疫), Metabolic (代谢), Metabolism (代谢), Lipid (脂质 - 代谢相关)
# 保留了：原来的心血管和炎症关键词
keywords <- "Heart|Cardi|Atrial|Vascular|Artery|Coronary|Myocardial|Fibrosis|Inflammation|Failure|Remodeling|Immune|Metabolic|Metabolism|Lipid"

message(paste("正在筛选包含以下关键词的条目:", keywords))

# 4. 进行筛选
df_selected <- df_all %>%
  # ignore.case = TRUE 表示不区分大小写
  filter(grepl(keywords, Description, ignore.case = TRUE)) %>%
  # 按照 p.adjust (显著性) 从小到大排序
  arrange(p.adjust)

# 检查筛选出了多少条
message(paste("筛选后剩余条目数:", nrow(df_selected)))

if (nrow(df_selected) == 0) {
  warning("没有找到包含这些关键词的疾病！")
} else {
  
  # 5. 数据清洗：处理 GeneRatio
  df_selected$GeneRatio_Num <- sapply(df_selected$GeneRatio, function(x) {
    nums <- as.numeric(unlist(strsplit(x, "/")))
    return(nums[1] / nums[2])
  })
  
  # 6. 只取前 20 个最显著的 (因为类别多了，展示数量可以稍微增加一点)
  top_n_plot <- 20
  if (nrow(df_selected) > top_n_plot) {
    df_plot <- df_selected[1:top_n_plot, ]
  } else {
    df_plot <- df_selected
  }
  
  # 7. 绘图
  p_custom <- ggplot(df_plot, aes(x = GeneRatio_Num, 
                                  y = reorder(Description, GeneRatio_Num))) + 
    geom_point(aes(size = Count, color = p.adjust)) +
    
    scale_color_gradient(low = "#E41A1C", high = "#377EB8", 
                         name = "p.adjust", 
                         guide = guide_colorbar(reverse = TRUE)) +
    
    labs(title = "Selected Diseases (CV, Immune, Metabolic)",
         subtitle = paste("Top", nrow(df_plot), "filtered terms from DisGeNET"),
         x = "GeneRatio", 
         y = NULL) +
    
    theme_bw() +
    theme(axis.text.y = element_text(size = 11, color = "black"),
          axis.text.x = element_text(size = 10),
          plot.title = element_text(size = 14, face = "bold"))
  
  print(p_custom)
  
  # 8. 保存图片
  ggsave("C:\\Users\\Administrator\\Desktop\\Selected_DisGeNET_All_Categories.pdf", 
         p_custom, width = 10, height = 8) # 稍微调高一点高度
  message("包含四大类筛选的气泡图已保存！")
}

