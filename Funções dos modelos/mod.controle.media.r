mod.controle.media <- function(tab, comp.nome = NULL, plot = FALSE) {
  
  # Criar o nome do arquivo dinamicamente
  arquivo_saida <- paste0("prev.mod.controle.media", comp.nome, ".csv")
  
  # Verificar se o arquivo já existe
  arquivo_existe <- file.exists(arquivo_saida)
  
  # Validar estrutura do dataset
  if (!all(c("id", "ano", "rodada", "mand", "vist", "plac_mand", "plac_vist") %in% colnames(tab))) {
    stop("O dataset 'tab' deve conter as colunas: id, ano, rodada, mand, vist, plac_mand, plac_vist.")
  }
  
  # Ajustar o dataset para incluir índices
  tab1 <- tab
  tab1$indice <- seq_len(nrow(tab1))
  
  # Adicionar o primeiro jogo
  prev <- data.frame(
    id        = tab1$id[1],
    ano       = tab1$ano[1],
    rodada    = tab1$rodada[1],
    mandante  = tab1$mand[1],
    visitante = tab1$vist[1],
    golman   = 0,
    golvis   = 0,
    resultado = "em"
  )
  
  # Escrever o primeiro registro
  write.table(
    prev,
    file = arquivo_saida,
    sep = ",",
    row.names = FALSE,
    col.names = !arquivo_existe,
    append = arquivo_existe
  )
  
  # Atualizar status do arquivo
  arquivo_existe <- TRUE
  
  # Loop para processar as rodadas
  for (n in 2:nrow(tab1)) {
    cat("Processando jogo", n, "de", nrow(tab1), "\n")
    
    # Dados anteriores à rodada atual
    dados <- subset(tab1, indice <= n - 1)
    times <- sort(union(dados$mand, dados$vist))
    
    # Média de gols por time
    gols <- data.frame(
      time = c(dados$mand, dados$vist),
      gols = c(dados$plac_mand, dados$plac_vist)
    )
    media_gols <- merge(
      data.frame(time = times),
      aggregate(gols ~ time, data = gols, FUN = mean),
      by = "time",
      all.x = TRUE
    )
    media_gols[is.na(media_gols$gols), "gols"] <- 0
    
    # Dados da rodada atual
    dados2 <- subset(tab1, indice == n)
    man2 <- dados2$mand
    vis2 <- dados2$vist
    
    # Previsões
    lambda1 <- ifelse(any(media_gols$time == man2), 
                      round(media_gols[media_gols$time == man2, 2]), 
                      0)
    lambda2 <- ifelse(any(media_gols$time == vis2), 
                      round(media_gols[media_gols$time == vis2, 2]), 
                      0)
    resul <- ifelse(lambda1 > lambda2, "vm", 
                    ifelse(lambda1 < lambda2, "vv", "em"))
    
    # Registro da rodada
    prev <- data.frame(
      id        = dados2$id,
      ano       = dados2$ano,
      rodada    = dados2$rodada,
      mandante  = dados2$mand,
      visitante = dados2$vist,
      golman   = lambda1,
      golvis   = lambda2,
      resultado = resul
    )
    
    if (plot) {
      print(prev)
    }
    
    # Escrever no arquivo
    write.table(
      prev,
      file = arquivo_saida,
      sep = ",",
      row.names = FALSE,
      col.names = !arquivo_existe,
      append = arquivo_existe
    )
  }
  
  cat("Data frame", arquivo_saida, "de previsões criado e salvo na pasta.\n")
}

