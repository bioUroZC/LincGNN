#=======================================================
# lncRNA annotation and sequence extraction using biomaRt
#=======================================================

rm(list = ls())

library(biomaRt)

setwd("D:\\DuanWork\\S2Sec\\3OSmodel\\year1\\NodePre")

ensembl = useMart("ensembl", dataset = "hsapiens_gene_ensembl")

lnc <- read.csv('Label.csv', row.names = 1)
ids = lnc$Regulator
ids[1:10]

ids <- ids[!is.na(ids) & ids != ""]
ids <- gsub(" ", "", ids)

# 4. Query Ensembl gene IDs for each symbol
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

#Merge symbol/gene_id/transcript_id/biotype/sequence information
final <- merge(lnc_trans, seqs, by.x = "ensembl_transcript_id", by.y = "ensembl_transcript_id")
final <- merge(info, final, by = "ensembl_gene_id", all.y = TRUE)
head(final)

# Add a column for cDNA sequence length, and keep the longest transcript per gene
library(dplyr)
final_unique <- final %>%
  mutate(seq_length = nchar(cdna)) %>%
  group_by(ensembl_gene_id) %>%
  slice_max(seq_length, n = 1, with_ties = FALSE) %>%
  ungroup()

# Preview and save the final result
head(final_unique)
write.csv(final_unique, file = 'lncrna_anno.csv')
