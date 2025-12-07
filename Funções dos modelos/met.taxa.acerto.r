met.taxa.acerto <- function(previsto, observado, retornar_comparacao = FALSE) {
  # Verifica se os vetores têm o mesmo comprimento
  if (length(previsto) != length(observado)) {
    stop("Os vetores devem ter o mesmo comprimento.")
  }
  
  # Remove valores NA
  tabela <- na.omit(data.frame(previsto = previsto,
                               observado = observado))
  
  # Cria o vetor de comparação
  vetor_comparacao <- as.integer(tabela$previsto == tabela$observado)
  
  # Calcula a taxa de acerto
  taxa_acerto <- mean(vetor_comparacao)
  
  # Retorna comparação detalhada se necessário
  if (retornar_comparacao) {
    comparacao <- cbind(tabela, acerto = vetor_comparacao)
    return(list(taxa_acerto = taxa_acerto, comparacao = comparacao))
  }
  
  return(taxa_acerto)
}
