#=======================================================

#=======================================================

rm(list = ls())
library(dplyr)
library(readxl)
library(biomaRt)
library(ggplot2)
library(Biostrings)
library(officer)
library(flextable)
#=======================================================

#=======================================================

setwd("D:\\mywork\\lincGNN\\cal")

table2 <- read_excel("Table2_final_merged1.xlsx")
table2 <- as.data.frame(table2)
head(table2)

table2 <- table2 %>%
  dplyr::select(Regulator, cell_proliferation)

head(table2)

table2_unique <- table2 %>%
  group_by(Regulator) %>%
  summarise(across(everything(), max))
names(table2_unique)[2] <- 'cell_proliferation_label'

# table2_unique <- table2 %>%
#   group_by(Regulator) %>%
#   summarise(cell_proliferation_label = as.integer(names(sort(table(cell_proliferation), decreasing = TRUE)[1])))

table(table2_unique$cell_proliferation_label)


#=======================================================

#=======================================================

ensembl = useMart("ensembl", dataset = "hsapiens_gene_ensembl")
ids = table2_unique$Regulator
ids[1:10]

ids <- ids[!is.na(ids) & ids != ""]
ids <- gsub(" ", "", ids)

result_list <- list()
success_idx <- c()

for(j in seq_along(ids)) {
  cat("Testing:", ids[j], "\n")
  tryCatch({
    res <- getBM(
      attributes = c('hgnc_symbol', 'ensembl_gene_id'),
      filters = 'hgnc_symbol',
      values = ids[j],
      mart = ensembl
    )
    if(nrow(res) > 0){
      result_list[[length(result_list) + 1]] <- res
      success_idx <- c(success_idx, j)
      cat("Success:", ids[j], "\n")
    } else {
      cat("No result for:", ids[j], "\n")
    }
  }, error=function(e){
    cat("Error with:", ids[j], "\n")
  })
}

info <- do.call(rbind, result_list)
gene_ids <- unique(info$ensembl_gene_id)

#Query all transcripts and their biotypes for these genes
trans_info <- getBM(
  attributes = c('ensembl_gene_id', 'ensembl_transcript_id', 'transcript_biotype'),
  filters = 'ensembl_gene_id',
  values = gene_ids,
  mart = ensembl
)

table(trans_info$transcript_biotype)

# Keep only common lncRNA-related biotypes
lnc_biotype <- c(
  "lncRNA", "antisense", "sense_intronic", "sense_overlapping", 
  "processed_transcript", "3prime_overlapping_ncRNA", 
  "bidirectional_promoter_lncRNA", "non_coding", "macro_lncRNA"
)
lnc_trans <- subset(trans_info, transcript_biotype %in% lnc_biotype)

# retrieve cDNA sequences for transcript IDs
seqs <- getSequence(
  id = lnc_trans$ensembl_transcript_id,
  type = "ensembl_transcript_id",
  seqType = "cdna",
  mart = ensembl
)


data_ensemble <- merge(lnc_trans, seqs, by = "ensembl_transcript_id")
final <- merge(info, data_ensemble, by = "ensembl_gene_id")
head(final)

names(final)

final_unique <- final %>%
  mutate(seq_length = nchar(cdna)) %>%
  group_by(hgnc_symbol) %>%
  slice_max(seq_length, n = 1, with_ties = FALSE) %>%
  ungroup()

head(final_unique)

#=======================================================

#=======================================================

final_unique <- final_unique %>%
  dplyr::select(hgnc_symbol, cdna)


final_unique <- final_unique %>%
  mutate(
    length = nchar(cdna),
    GC_content = (stringr::str_count(cdna, "G") + stringr::str_count(cdna, "C")) / length,
    A_freq = stringr::str_count(cdna, "A") / length,
    T_freq = stringr::str_count(cdna, "T") / length,
    G_freq = stringr::str_count(cdna, "G") / length,
    C_freq = stringr::str_count(cdna, "C") / length
  )

# k-mer
seqs <- DNAStringSet(final_unique$cdna)

# 2-mer
oligo2 <- as.data.frame(oligonucleotideFrequency(seqs, width = 2))
colnames(oligo2) <- paste0("dimer_", colnames(oligo2))

# 3-mer
oligo3 <- as.data.frame(oligonucleotideFrequency(seqs, width = 3))
colnames(oligo3) <- paste0("trimer_", colnames(oligo3))

# 4-mer
oligo4 <- as.data.frame(oligonucleotideFrequency(seqs, width = 4))
colnames(oligo4) <- paste0("tetramer_", colnames(oligo4))


features <- bind_cols(final_unique, oligo2, oligo3, oligo4)
features$cdna <- NULL

names(features)[1] <- 'Regulator'
features[1:5,1:5]
names(features)

min(features$length)
max(features$length)



basic_features <- c("length", "GC_content", "A_freq", "T_freq", "G_freq", "C_freq")

dt_summary <- data.frame(
  kmer_type = c("basic", "dimer", "trimer", "tetramer"),
  feature_names = c(
    paste(basic_features, collapse = ", "),
    paste(names(oligo2), collapse = ", "),
    paste(names(oligo3), collapse = ", "),
    paste(names(oligo4), collapse = ", ")
  ),
  stringsAsFactors = FALSE
)

dt_summary[1:2,]

dt_summary$feature_names <- gsub('dimer_', '',dt_summary$feature_names)
dt_summary$feature_names <- gsub('trimer_', '',dt_summary$feature_names)
dt_summary$feature_names <- gsub('tetramer_', '',dt_summary$feature_names)


ft <- flextable(dt_summary)
doc <- read_docx() %>% body_add_flextable(ft)
print(doc, target = "dt2_summary.docx")


#=======================================================

#=======================================================

Label_final <- merge(table2_unique, features, by='Regulator')

rownames(Label_final) <- NULL


Net <- read_excel("Table1_final.xlsx")
Net <- as.data.frame(Net)
table(Net$RegulatorType)

Net_final1 <- Net[which(Net$RegulatorType == 'lncRNA'),]
Net_final2 <- Net[which(Net$RegulatorType == 'PCG'),]

Net_final1 <- Net_final1[which(Net_final1$Regulator %in% Label_final$Regulator),]

Net_final <- rbind(Net_final1, Net_final2)


rownames(Net_final) <- NULL

write.csv(Net_final, file = "Net_final.csv")

write.csv(Label_final, file = "Label_final.csv")


#=======================================================

#=======================================================

counts <- as.data.frame(table(Label_final$cell_proliferation_label))
colnames(counts) <- c("label", "Freq")


lancet_colors <- c("0" = "#0072B2", "1" = "#009E73") 

p <- ggplot(counts, aes(x = label, y = Freq, fill = label)) +
  geom_bar(stat = "identity", width = 0.6, color = "black", size = 1) +     
  geom_text(aes(label = Freq), vjust = -0.5, size = 6) +                  
  scale_fill_manual(values = lancet_colors) +                               
  labs(x = "Cell Proliferation Label", y = "Count") +
  theme_bw(base_size = 16) +                                                 
  theme(
    panel.grid.major = element_blank(),  
    panel.grid.minor = element_blank(),   
    panel.border = element_rect(color = "black", size = 1),  
    legend.position = "none"              
  )

print(p)

pdf(file = 'Label_number.pdf', height = 5, width = 5)
print(p)
dev.off()

