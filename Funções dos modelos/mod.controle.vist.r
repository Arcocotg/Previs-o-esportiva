mod.controle.vist <- function(tab, comp.nome = NULL, plot = F) {
  
  # Criar o nome do arquivo dinamicamente
  arquivo_saida <- paste0("prev.mod.controle.vist", comp.nome, ".csv")
  
  # Verificar se o arquivo já existe
  arquivo_existe <- file.exists(arquivo_saida)
  
  for (i in 1:nrow(tab)) {
    jogo2 <- tab[i, ]
    
    prev <- data.frame(
      id        = jogo2$id,
      ano       = jogo2$ano,
      rodada    = jogo2$rodada,
      mandante  = jogo2$mand,
      visitante = jogo2$vist,
      pvm       = 0,
      pe        = 0,
      pvv       = 1,
      resultado = "vv"
    )
    
    if (plot == T) {
      print(prev)
    }
    
    # Escrever no arquivo com cabeçalho apenas na primeira vez
    write_csv(
      prev,
      arquivo_saida,
      na = "NA",
      append = arquivo_existe # Escreve cabeçalho apenas se o arquivo não existe
    )
    
    # Atualizar o status do arquivo para as próximas iterações
    arquivo_existe <- TRUE
  }
  
  cat("Data frame", arquivo_saida, "de previsões criado e salvo na pasta.\n")
}





