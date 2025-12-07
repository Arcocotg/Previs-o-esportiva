met.epmp <- function(previsto, observado, pvm, pem, pvv, retornar_comparacao = FALSE) {
  # Verifica se os vetores têm o mesmo comprimento
  if (!all(length(previsto) == length(observado), 
           length(pvm) == length(observado), 
           length(pem) == length(observado), 
           length(pvv) == length(observado))) {
    stop("Todos os vetores devem ter o mesmo comprimento.")
  }
  
  # Remove valores NA
  tabela <- na.omit(data.frame(previsto = previsto,
                               observado = observado,
                               pvm = pvm,
                               pem = pem,
                               pvv = pvv))
  
  # Criar vetor de comparação (1 = acerto, 0 = erro)
  vetor_comparacao <- as.integer(tabela$previsto == tabela$observado)
  
  # Cálculo corrigido do erro de predição ponderado
  erro_pred_pond <- pmax(tabela$pvm, tabela$pem, tabela$pvv) * (1 - vetor_comparacao)
  
  # Taxa de acerto
  taxa_acerto <- mean(vetor_comparacao)
  
  # Erro médio ponderado
  epmp <- mean(erro_pred_pond)
  
  # Retornar comparação se solicitado
  if (retornar_comparacao) {
    comparacao <- cbind(tabela, acerto = vetor_comparacao, erro_pred_pond = erro_pred_pond)
    return(list(
      taxa_acerto = taxa_acerto,
      epmp = epmp,
      comparacao = comparacao
    ))
  }
  
  # Retorna apenas o erro médio ponderado
  return(epmp)
}
