#out
genes <- c('Met', 'Popdc3', 'Lmo1', 'Blank-13', 'Fstl5', 'Adgrv1', 'Kcng2', 'Npy2r', 'Nr2f1', 'C1ql1', 'Rwdd2a', 'Nts', 'Eomes', 'Cidea', 'Stac', 'Adamtsl1', 'Blank-33', 'Ust', 'Rgs4', 'Gpr101', 'Ndst4', 'G630016G05Rik', 'Trpc7', 'Rorb', 'Tent5a', 'Dsg2', 'Svep1')


##in
#genes <- c('Prom1', 'Grik1', 'Pitx2', 'Syndig1', 'Lamp5', 'Rgs4', 'Blank-32', 'Gpr139', 'Pparg', 'Bmp4', 'Rspo1', 'Plpp4', 'Blank-13', 'Blank-7', 'Rorb', 'Dsg2', 'Grm3')


library(clusterProfiler)
library(org.Mm.eg.db)
library(GOplot)
library(dplyr)
library(tidyr)
library(RColorBrewer)
library(stringr)

## ======================
## 1. Gene list
## ======================

genes <- c('Prom1', 'Grik1', 'Pitx2', 'Syndig1', 'Lamp5', 'Rgs4', 'Blank-32', 'Gpr139', 'Pparg', 'Bmp4', 'Rspo1', 'Plpp4', 'Blank-13', 'Blank-7', 'Rorb', 'Dsg2', 'Grm3')

## ======================
## 2. ID conversion
## ======================

gene_df <- bitr(
  genes,
  fromType = "SYMBOL",
  toType   = "ENTREZID",
  OrgDb    = org.Mm.eg.db
)

gene_entrez <- gene_df$ENTREZID

## ======================
## 3. GO enrichment
## ======================

ego <- enrichGO(
  gene          = gene_entrez,
  OrgDb         = org.Mm.eg.db,
  keyType       = "ENTREZID",
  ont           = "CC",
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.1,
  qvalueCutoff  = 0.2,
  readable      = TRUE
)

## ======================
## 4. Top terms + 星号（关键位置）
## ======================

go_show <- ego@result %>%
  arrange(pvalue) %>%
  slice(1:10) %>%
  select(Term = Description, geneID, pvalue) %>%
  mutate(
    signif = case_when(
      pvalue < 0.001 ~ "***",
      pvalue < 0.01  ~ "**",
      pvalue < 0.05  ~ "*",
      TRUE           ~ ""
    ),
    Term = paste0(str_trunc(Term, 35), " ", signif)
  )

## ======================
## 5. Expand genes
## ======================

go_clean <- go_show %>%
  separate_rows(geneID, sep = "/") %>%
  rename(Genes = geneID)

## ======================
## 6. 小梯度 logFC（你原始做法）
## ======================

set.seed(123)
go_clean$logFC <- runif(nrow(go_clean), -0.1, 0.1)

## ======================
## 7. GOChord input
## ======================

df_chord <- chord_dat(
  go_clean,
  process = unique(go_clean$Term)
)

df_chord <- as.data.frame(df_chord)
df_chord$logFC <- go_clean$logFC[
  match(rownames(df_chord), go_clean$Genes)
]

## ======================
## 8. Colors（完全按你的方式）
## ======================

term_colors <- brewer.pal(
  max(8, length(unique(go_clean$Term))),
  "Set3"
)

gene_color <- c("#E64B35", "#E64B35", "#E64B35")

## ======================
## 9. Plot
## ======================

par(pty = "s", xpd = NA)

GOChord(
  df_chord,
  space      = 0.02,
  lfc.col    = gene_color,
  ribbon.col = term_colors,
  gene.order = "alphabetical",
  gene.size  = 4,
  gene.space = 0.3
)
