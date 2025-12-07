mod.controle.implicito.media <- function(tab, 
                                         comp.nome = NULL, 
                                         plot = FALSE) {
  
  arquivo_saida <- paste0("prev.mod.controle.implicito.media", comp.nome, ".csv")
  arquivo_existe <- file.exists(arquivo_saida)
  
  tab1 <- tab
  tab1$indice <- seq_len(nrow(tab1))
  
  # ===== PREVISÃO DO PRIMEIRO JOGO =====
  prev <- data.frame(
    id        = tab1$id[1],
    ano       = tab1$ano[1],
    rodada    = tab1$rodada[1],
    mandante  = tab1$mand[1],
    visitante = tab1$vist[1],
    pvm       = 0,
    pe        = 1,
    pvv       = 0,
    golman    = 0,
    golvis    = 0,
    lambda1   = 0,
    lambda2   = 0,
    resultado = "em"
  )
  
  write_csv(prev, arquivo_saida, na = "NA", append = arquivo_existe)
  arquivo_existe <- TRUE
  
  # ===== LOOP =====
  for (n in 2:nrow(tab1)) {
    
    dados <- tab1[tab1$indice <= n-1, ]
    dados2 <- tab1[tab1$indice == n, ]
    
    # Evitar erros se dados2 estiver vazio
    if (nrow(dados2) == 0) {
      message("Aviso: nenhum jogo encontrado para indice ", n)
      next
    }
    
    # Montar tabela de gols
    gols <- data.frame(
      time = c(dados$mand, dados$vist),
      gols = c(dados$plac_mand, dados$plac_vist)
    )
    
    media_gols <- aggregate(gols ~ time, data = gols, FUN = mean)
    
    man2 <- dados2$mand
    vis2 <- dados2$vist
    
    # Caso um time ainda não tenha histórico, usa média geral
    media_global <- mean(gols$gols)
    
    lambda1 <- ifelse(man2 %in% media_gols$time,
                      media_gols$gols[media_gols$time == man2],
                      media_global)
    
    lambda2 <- ifelse(vis2 %in% media_gols$time,
                      media_gols$gols[media_gols$time == vis2],
                      media_global)
    
    # Matriz de Holgate
    pb1 <- dpois(0:7, lambda1)
    pb2 <- dpois(0:7, lambda2)
    mh  <- pb1 %*% t(pb2)
    
    VM <- sum(mh[lower.tri(mh)])
    ET <- sum(diag(mh))
    VV <- sum(mh[upper.tri(mh)])
    
    # Evita erro se pplacar for vazio
    placar <- max(mh, na.rm = TRUE)
    pplacar <- which(mh == placar, arr.ind = TRUE)
    
    if (nrow(pplacar) == 0) {
      golman <- NA
      golvis <- NA
    } else {
      golman <- pplacar[1,1] - 1
      golvis <- pplacar[1,2] - 1
    }
    
    prev <- data.frame(
      id        = dados2$id,
      ano       = dados2$ano,
      rodada    = dados2$rodada,
      mandante  = dados2$mand,
      visitante = dados2$vist,
      pvm       = VM,
      pe        = ET,
      pvv       = VV,
      golman    = golman,
      golvis    = golvis,
      lambda1   = lambda1,
      lambda2   = lambda2
    )
    
    prev$resultado <- ifelse(max(prev$pvm, prev$pe, prev$pvv) == prev$pvm, "vm",
                             ifelse(max(prev$pvm, prev$pe, prev$pvv) == prev$pvv, "vv", "em"))
    
    write_csv(prev, arquivo_saida, na = "NA", append = TRUE)
  }
  
  cat("Data frame", arquivo_saida, "de previsões criado e salvo.\n")
}