mod.arr.SD0.ano <- function(tab, comp.nome = NULL, plot = F) {
  
  # Criar o nome do arquivo dinamicamente
  arquivo_saida <- paste0("prev.mod.arr.SD0.ano", comp.nome, ".csv")
  
  # Verificar se o arquivo já existe
  arquivo_existe <- file.exists(arquivo_saida)
  
  
  # Ordenar tab pelo dia
  tab1 <- tab |> arrange(dia)
  
  # fazer a coluna semestre 
  
  tab1$semestre <- 0
  
  for (i in 1:nrow(tab)) {
    if(format(tab1$dia[i] , "%m") < "07" ){
      tab1$semestre[i] <- 1
    }
  }
  
  ### constroi o vetor indice 
  {
    tab_ano <- tab1 |>
      mutate(indice0 = NA)
    
    ano_atual <- tab_ano$ano[1]
    k=1
    
    
    for (j in 1:nrow(tab_ano)) {
      
      if(ano_atual == tab_ano$ano[j]){
        tab_ano$indice0[j] <- k
      }else{
        k <- k+1
        ano_atual <- tab_ano$ano[j]
        tab_ano$indice0[j] <- k
      }
    }
  }
  

  for (o in 1:length(unique(tab_ano$indice0))) {
    
    tab2 <- tab_ano[tab_ano$indice0 == o,]
  ########################################
  #Ajustando o banco de dados para calcular o parametro após cada jogo
  
  tab2$indice <- seq(1:nrow(tab2))
  
  #adicionar o primeiro jogo 
  {
    #### escrevendo os dados.
    prev <- data.frame(
      id        = tab2$id[1],
      ano       = tab2$ano[1],
      rodada    = tab2$rodada[1],
      mandante  = tab2$mand[1],
      visitante = tab2$vist[1],
      pvm       = 0,
      pe        = 1,
      pvv       = 0,
      golman    = 0,
      golvis    = 0,
      lambda1    = 1,
      lambda2    = 1,
      resultado = "em"
    )
    
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
  
  # faz as previsões 
  
  for (n in 2:nrow(tab2)){
    
     #n <- 25
    
    ###### jogos antes da previsão 
    
    dados <- subset(tab2, indice <= n-1)

    man <- unique(dados$mand)
    vis <- unique(dados$vist)

    times <- sort(union(man,vis))
    
    g1 <- dados$plac_mand
    g2 <- dados$plac_vist
    
    nj <- nrow(dados)
    nt <- length(times)
    
    ##########################
    ### Metodo SD 0
    ##########################
 
    matriz <- matrix(0,nj,nt+1)
    #Preencha a matriz 'matriz'
    for (i in 1:nj){
      t1 <- dados$mand[i]
      x1 <- which(times[]==t1)
      matriz[i,x1] <- 1
      t2 <- dados$vist[i]
      x2 <- which(times[]==t2)
      matriz[i,x2] <- -1
      if (is.na(dados$Camp[i])){matriz[i,nt+1] <- 1}
    }
    
    #Calculo da variavel peso
    
    dados$peso <- 0
    
    for (i in 1:nrow(dados)) {
      dados$peso[i] = 1.4+(0.4*((dados$dia_num[i] -(dados$semestre[i] *730/2))-dados$dia_num[nj])/730)
      
      if(dados$peso[i]<0){dados$peso[i] <- 0}
    }
 
    
    #Ajusta uma modelo de regress?o para 'd' como fun??o da matriz-1, 
    #com ponderado por w
    
    d <- g1-g2
    bd <- lm(d ~ matriz - 1, weights = dados$peso)
    s <- g1+g2
    #Toma o valor absoluto das entradas de 'matriz'
    matriz <- abs(matriz)
    #Ajusta uma modelo de regress?o para 's' como fun??o da matriz-1, 
    #com ponderado por w
    bs <- lm(s ~ matriz - 1, weights = dados$peso)
    #Coloca as palavras 'efeito casa' na posi??o nt+1 do vetor times
    times[nt+1] <- 'efeito casa'
    #Cria e preenche uma tabela chamada 'parametros1',
    #com os resultados da estimacao
    parametros1 <- data.frame(matrix(NA,(nt+1),3))
    parametros1[,1] <- times
    parametros1[,2] <- coef(bd)
    parametros1[is.na(coef(bd)),2] <- 0
    parametros1[,3] <- coef(bs)
    parametros1[is.na(coef(bs)),3] <- 0
    colnames(parametros1) <- c('Time','Betas (Dif)','Alfas (Soma)')
    parametros1
    
    # parametrosT[,,n] <- rbind(as.matrix(parametros1),matrix(NA,pnt-nrow(parametros1),ncol(parametros1)))
    
    ############################################################################
    ############################################################################
    #Previs?o para cada rodada do campeonato 
    
    
    dados2 <- subset(tab2, indice == n)
    
    man2 <- dados2$mand
    vis2 <- dados2$vist
    
    times2 <- sort(union(man2,vis2))
    
    nj2 <- dim(dados2)[1]
    nt2 <- length(times2)
    
    ##################################
    
    # construção da matriz
    
    matriz_beta <- times
    
    for (i in 1:nt) {
      if(times[i]==man2){matriz_beta[i] <- 1 }else{
        if(times[i]==vis2){matriz_beta[i] <- -1 }else{
          matriz_beta[i] <- 0
        }
      }
    }
    if (is.na(dados2$Camp)){matriz_beta[nt+1] <- 1}else{matriz_beta[nt+1] <- 0}
    
    matriz_beta <- as.numeric(matriz_beta)
    
    matriz_alpha<-abs(matriz_beta)
    
    
    ES <- matriz_alpha%*%parametros1[,3]
    ED <- matriz_beta%*%parametros1[,2]
    
    lambda1 <- abs((ES+ED)/2)
    lambda2 <- abs((ES-ED)/2)
    
    # Holgate feito por meio de duas poisson
    
    pb1<-dpois(0:7, lambda = lambda1) 
    pb2<-dpois(0:7, lambda = lambda2)
    mh<-pb1%*%t(pb2)
    
    ### vetor contendo matrizes 
    
    # soma da Matriz de Holgate
    VM <- round(sum(mh[lower.tri(mh)]),5)
    ET <- round(sum(diag(mh)),5)
    VV <- round(sum(mh[upper.tri(mh)]),5)
    
    
    placar <- max(mh)
    pplacar <- which(mh==placar, arr.ind = T)
    
    #VM+ET+VV #  essa soma ? igual a 1?
    #print(VM)
    #print(ET)
    #print(VV)
    
    #### escrevendo os dados.
    prev <- data.frame(
      id        = dados2$id,
      ano       = dados2$ano,
      rodada    = dados2$rodada,
      mandante  = dados2$mand,
      visitante = dados2$vist,
      pvm       = VM,
      pe        = ET,
      pvv       = VV,
      golman    = (pplacar[1]-1),
      golvis    = (pplacar[2]-1),
      lambda1    = lambda1,
      lambda2    = lambda2
    )
    
    prev$resultado <- ifelse(max(prev$pvm,prev$pe,prev$pvv) == prev$pvm, "vm", 
                             ifelse(max(prev$pvm,prev$pe,prev$pvv) == prev$pvv, "vv", "em"))
    if(plot){print(prev)}
    
    # Escrever no arquivo com cabeçalho apenas na primeira vez
    write_csv(
      prev,
      arquivo_saida,
      na = "NA",
      append = arquivo_existe # Escreve cabeçalho apenas se o arquivo não existe
    )
  } 
}
  cat("Data frame", arquivo_saida, "de previsões criado e salvo na pasta.\n")
}

