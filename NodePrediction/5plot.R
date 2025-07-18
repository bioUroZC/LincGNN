#=======================================================

#=======================================================

rm(list = ls())
library(dplyr)
library(ggplot2)
library(reshape2)

setwd("D:\\mywork\\lincGNN\\cal")

#=======================================================

#=======================================================

data <- data.frame(FeatureType=c('Sequence-based', 'Network-based'),
                   Number=c(342, 137))

data$FeatureType <- factor(data$FeatureType, levels = c('Sequence-based', 'Network-based'))

lancet_colors <- c("Sequence-based" = "#3182bd", "Network-based" = "#e6550d")


p <- ggplot(data, aes(x = FeatureType, y = Number, fill = FeatureType)) +
  geom_bar(stat = "identity", width = 0.6, color = "black", size = 1) +     
  geom_text(aes(label = Number), vjust = -0.5, size = 6) +                  
  scale_fill_manual(values = lancet_colors) +                               
  labs(y = "Count") +
  theme_bw(base_size = 16) +                                                 
  theme(
    panel.grid.major = element_blank(),  
    panel.grid.minor = element_blank(),   
    panel.border = element_rect(color = "black", size = 1),  
    legend.position = "none"              
  )

print(p)

pdf(file = 'Feature_Value.pdf', height = 5, width = 6)
print(p)
dev.off()

#=======================================================

#=======================================================


ml <- read.csv('ml_metrics.csv', header = TRUE)
names(ml)[2] <- "SVM"
han <- read.csv('han_metrics.csv', header = TRUE)
names(han)[2] <- "HAN"

data <- merge(ml, han, by = 'metric')
data$metric <- gsub('balanced_accuracy', 'bal_acc', data$metric)

nice_metric <- c(
  bal_acc   = "Accuracy",
  precision = "Precision",
  recall    = "Recall",
  F1_score  = "F1Score",
  ROC_AUC   = "AUC"
)

df_long <- reshape2::melt(data, id.vars = "metric", variable.name = "Model", value.name = "Value")
df_long$metric_nice <- factor(nice_metric[as.character(df_long$metric)],
                              levels = nice_metric)

p <- ggplot(df_long, aes(x = metric_nice, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", Value)), 
            position = position_dodge(width = 0.8), vjust = -0.2, size = 4) +
  scale_fill_manual(values = c("#3182bd", "#e6550d")) +
  labs(x = NULL, y = "Value", fill = "Model") +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA, size = 1),
    axis.text.x = element_text(angle = 45, hjust = 1, face = "italic")
  )

print(p)

pdf(file = 'Performance.pdf', height = 5, width = 6)
print(p)
dev.off()

#=======================================================

#=======================================================



cm <- data.frame(
  Pred = rep(c("Negative", "Positive"), times=2),
  True = rep(c("Negative", "Positive"), each=2),
  Count = c(10, 19, 19, 73)
)

cm

cm$Pred <- factor(cm$Pred, levels = )


p <- ggplot(cm, aes(x = Pred, y = True, fill = Count)) +
  geom_tile(color = "white", width=0.95, height=0.95) +
  geom_text(aes(label = Count), size = 8, color = ifelse(cm$Count > 20, "white", "black"), fontface = "bold") +
  scale_fill_gradient(low = "#deebf7", high = "#08306b") +
  labs(
    title = "Confusion Matrix (Test Set)",
    y = "True Label",
    x = "Predicted Label"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(hjust = 0.5, face="bold"),
    axis.title.x = element_text(face="bold"),
    axis.title.y = element_text(face="bold"),
    panel.grid = element_blank()
  )

print(p)


pdf(file = 'ml_cm.pdf', height = 5, width = 5)
print(p)
dev.off()



#=======================================================

#=======================================================



cm <- data.frame(
  Pred = rep(c("Negative", "Positive"), times=2),
  True = rep(c("Negative", "Positive"), each=2),
  Count = c(12, 14, 12, 83)
)

cm

cm$Pred <- factor(cm$Pred, levels = )


p <- ggplot(cm, aes(x = Pred, y = True, fill = Count)) +
  geom_tile(color = "white", width=0.95, height=0.95) +
  geom_text(aes(label = Count), size = 8, color = ifelse(cm$Count > 20, "white", "black"), fontface = "bold") +
  scale_fill_gradient(low = "#deebf7", high = "#08306b") +
  labs(
    title = "Confusion Matrix (Test Set)",
    y = "True Label",
    x = "Predicted Label"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(hjust = 0.5, face="bold"),
    axis.title.x = element_text(face="bold"),
    axis.title.y = element_text(face="bold"),
    panel.grid = element_blank()
  )

print(p)


pdf(file = 'han_cm.pdf', height = 5, width = 5)
print(p)
dev.off()