mod.controle.uniforme <- function(tab, comp.nome = NULL, plot = F) {
  
  # Criar o nome do arquivo dinamicamente
  arquivo_saida <- paste0("prev.mod.controle.uniforme", comp.nome, ".csv")
  
  # Verificar se o arquivo já existe
  arquivo_existe <- file.exists(arquivo_saida)
  
  # Previsão
  for (i in 1:nrow(tab)) {
    jogo <- tab[i, ]
    
    previsão <- sort(runif(2))
    
    prev <- data.frame(
      id        = jogo$id,
      ano       = jogo$ano,
      rodada    = jogo$rodada,
      mandante  = jogo$mand,
      visitante = jogo$vist,
      pvm       = previsão[1],
      pe        = (previsão[2] - previsão[1]),
      pvv       = 1 - previsão[2]
    )
    
    prev$resultado <- ifelse(
      max(prev$pvm, prev$pe, prev$pvv) == prev$pvm, "vm",
      ifelse(max(prev$pvm, prev$pe, prev$pvv) == prev$pvv, "vv", "em")
    )
    
    if (plot == T) {
      print(prev)
    }
    
    # Salvar o arquivo com cabeçalho apenas se for a primeira vez
    write_csv(
      prev,
      arquivo_saida,
      na = "NA",
      append = arquivo_existe, # Controla a inclusão do cabeçalho
      col_names = !arquivo_existe # Inclui cabeçalho apenas se o arquivo ainda não existe
    )
    
    # Atualizar o status do arquivo para as próximas iterações
    arquivo_existe <- TRUE
  }
  
  cat("Data frame", arquivo_saida, "de previsões criado e salvo na pasta.\n")
}
