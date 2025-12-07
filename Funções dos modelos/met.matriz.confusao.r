met.cm <- function(previsto, observado, retornar_comparacao = FALSE) {
  require(caret)
  
  if (length(previsto) != length(observado)) {
    stop("Comprimentos diferentes entre previsto e observado.")
  }
  
  tabela <- na.omit(data.frame(
    previsto  = factor(previsto),
    observado = factor(observado)
  ))
  
  # Nivelar fatores
  tabela$observado <- factor(tabela$observado, levels = levels(tabela$previsto))
  
  cm <- confusionMatrix(tabela$previsto, tabela$observado)
  
  # ---------------------------------------------------
  # DETECTA AUTOMATICAMENTE BINÁRIO OU MULTICLASS
  # ---------------------------------------------------
  byClass <- cm$byClass
  
  if (is.null(dim(byClass))) {
    # ----------- BINÁRIO -----------
    precision_per_class <- byClass["Pos Pred Value"]
    recall_per_class    <- byClass["Sensitivity"]
    f1_per_class <- 2 * precision_per_class * recall_per_class /
      (precision_per_class + recall_per_class)
    
    classes <- levels(tabela$observado)
    
  } else {
    # ----------- MULTICLASS -----------
    precision_per_class <- byClass[, "Pos Pred Value"]
    recall_per_class    <- byClass[, "Sensitivity"]
    f1_per_class <- 2 * precision_per_class * recall_per_class /
      (precision_per_class + recall_per_class)
    
    classes <- rownames(byClass)
  }
  
  # -----------------
  # MCC MULTICLASS
  # -----------------
  mat <- cm$table
  n <- sum(mat)
  row_sum <- rowSums(mat)
  col_sum <- colSums(mat)
  trace_term <- sum(diag(mat))
  Accuracy <- cm$overall["Accuracy"]
  
  mcc_multiclass <- (trace_term * n - sum(row_sum * col_sum)) /
    sqrt((n^2 - sum(row_sum^2)) * (n^2 - sum(col_sum^2)))
  
  # MÉTRICAS AGREGADAS
  macro_precision <- mean(precision_per_class, na.rm = TRUE)
  macro_recall    <- mean(recall_per_class,    na.rm = TRUE)
  macro_f1        <- mean(f1_per_class,        na.rm = TRUE)
  
  weighted_precision <- sum(precision_per_class * col_sum / n, na.rm = TRUE)
  weighted_recall    <- sum(recall_per_class    * col_sum / n, na.rm = TRUE)
  weighted_f1        <- sum(f1_per_class        * col_sum / n, na.rm = TRUE)
  
  # TABELA FINAL
  cm_resul <- data.frame(
    classe              = classes,
    Precision           = precision_per_class,
    Recall              = recall_per_class,
    F1                  = f1_per_class,
    Precision_macro     = macro_precision,
    Recall_macro        = macro_recall,
    F1_macro            = macro_f1,
    Precision_weighted  = weighted_precision,
    Recall_weighted     = weighted_recall,
    F1_weighted         = weighted_f1,
    Accuracy            = Accuracy,
    MCC                 = mcc_multiclass
  )
  
  if (retornar_comparacao){ 
    return(cm_resul) 
  }else{
    return(list(
      Accuracy        = Accuracy,
      Precision       = macro_precision,
      Recall          = macro_recall,
      F1              = macro_f1,
      MCC             = mcc_multiclass
    ))
    }
  
}