
prep_dados <- function(id,dia,ano,rodada,mand,vist,plac_mand,plac_vist,classicos){
  
  #filtrando quais jogos são considerados classicos
  classicos <- subset(classicos, classicos$`Considerar?`=="s")
  
  # primeiro verificando se o jogo é classico ou não
  Camp <- vector("character", length(mand))
  
  for (i in seq_along(mand)) {
    # Verificar se o par mandante-visitante ou visitante-mandante está na tabela de clássicos
    match_classico <- classicos$Classico[
      (mand[i] == classicos$Time1 & vist[i] == classicos$Time2) |
        (mand[i] == classicos$Time2 & vist[i] == classicos$Time1)
    ]
    
    if (length(match_classico) > 0) {
      Camp[i] <- "c"  # "c" indica clássico
    } else {
      Camp[i] <- NA   # NA para não clássico
    }
  }
  
  # definir o resultado final da partida 
  resul <- ifelse(plac_mand > plac_vist, "vm", ifelse(plac_mand == plac_vist, "em", "vv"))
  
  # definir data numérica para modelos que usam peso
  dia_num <- as.numeric(dia - as.Date("1900-01-01")) + 2
  
  # contruindo a tabela
  tab <- data.frame(id,
                    dia,
                    dia_num,
                    ano,
                    rodada,
                    mand,
                    vist,
                    plac_mand,
                    plac_vist,
                    resul,
                    Camp
                    )
  # Salvando a tabela
  assign("tab", tab, envir = .GlobalEnv)
  cat("Data frame tab criado e salvo na memória.\n")
  
  # filtrando  e organizando times
  times <- sort(unique(c(vist, mand)))
  
  #contruindo a tabela status
  times_status <- data.frame(
    Times = times, 
    pvm   = rep(0.4, length(times)), # prob vit mand
    pem   = rep(0.3, length(times)), # prob emp mand
    pdm   = rep(0.3, length(times)), # prob der mand
    pdv   = rep(0.4, length(times)), # prob vit vist
    pev   = rep(0.3, length(times)), # prob emp vist
    pvv   = rep(0.3, length(times)), # prob der vist
    Rc    = 0,   # pontos conquistados
    Rd    = 0,   # pontos disputados
    R     = 0    # rendimento = pontos conquistados/ pontos disputados
  )
  times_status0 <- times_status
  # Salvando a tabela de status de cada time
  assign("times_status", times_status, envir = .GlobalEnv)
  assign("times_status0", times_status0, envir = .GlobalEnv)
  cat("Data frame times_status criado e salvo na memória.\n")
 
}










