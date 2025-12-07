met.definetti <- function(pvm, pem, pvv, resul, retornar_componentes = FALSE) {
  # Remove valores NA
  tabela <- na.omit(data.frame(pvm = pvm, pem = pem, pvv = pvv, resul = resul))
  
  # Verifica se os vetores têm o mesmo comprimento
  if (!all(lengths(tabela) == nrow(tabela))) {
    stop("Todos os vetores devem ter o mesmo comprimento.")
  }
  
  # Converte resul para caracteres limpos
  tabela$resul <- trimws(tolower(as.character(tabela$resul)))
  
  # Verifica se 'resul' contém apenas valores válidos
  if (!all(tabela$resul %in% c("vm", "em", "vv"))) {
    stop("Os valores de 'resul' devem ser 'vm', 'em' ou 'vv'.")
  }
  
  # Cálculo vetorizado de DF
  valores_DF <- (tabela$pvm - as.integer(tabela$resul == "vm"))^2 + 
                (tabela$pem - as.integer(tabela$resul == "em"))^2 + 
                (tabela$pvv - as.integer(tabela$resul == "vv"))^2
  
  # Adiciona os valores DF na tabela
  tabela$DF <- valores_DF
  
  # Calcula a média dos valores de DF
  media_DF <- mean(valores_DF)
  
  # Retorna a média e a tabela de componentes, se solicitado
  if (retornar_componentes) {
    return(list(
      media_DF = media_DF,
      componentes = tabela
    ))
  }
  
  # Retorna apenas a média dos valores de DF
  return(media_DF)
}
