mod.arr.chance1.rodada <- function(tab, comp.nome = NULL, nr = 2, plot = F) {
  
  # Criar o nome do arquivo dinamicamente
  arquivo_saida <- paste0("prev.mod.arr.chance1.rodada", comp.nome, ".csv")
  
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
  
  # Ordenar tab pela rodada e ano
  tab1 <- tab1 |> arrange(ano, rodada)
  
  ### constroi o vetor indice 
  {
    tab_rod <- tab1 |>
      mutate(rod_ano = paste(ano,rodada)) |>
      mutate(indice0 = NA)
    
    rod_ano_atual <- tab_rod$rod_ano[1]
    k=1
    
    
    for (j in 1:nrow(tab_rod)) {
      
      if(rod_ano_atual == tab_rod$rod_ano[j]){
        tab_rod$indice0[j] <- k
      }else{
        k <- k+1
        rod_ano_atual <- tab_rod$rod_ano[j]
        tab_rod$indice0[j] <- k
      }
    }
    
    #estabelecer o vetor times
    times0 <- sort(union(tab_rod$mand,tab_rod$vist))
    
  }
  
  #fazer previsões para os primeiros jogos
  {
    tab2 <- tab_rod[tab_rod$indice0 <= nr,]
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
        lambda1    = 0,
        lambda2    = 0,
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
      
      #n <- 15
      
      ###### jogos antes da previsão 
      
      dados <- subset(tab2, indice <= n-1)
      
      man <- unique(dados$mand)
      vis <- unique(dados$vist)
      
      times <- times0
      
      g1 <- dados$plac_mand
      g2 <- dados$plac_vist
      
      nj <- nrow(dados)
      nt <- length(times)
      
      #Calculo da variavel peso
      {
        dados$peso <- 0
        
        for (i in 1:nrow(dados)) {
          dados$peso[i] = 1.4+(0.4*((dados$dia_num[i] -(dados$semestre[i] *730/2))-dados$dia_num[nj])/730)
          
          if(dados$peso[i]<0){dados$peso[i] <- 0}
        }
      }
      ##############################
      ### Método Chance I
      ##############################
      
      matriz <- matrix(0,(2*nj),((2*nt)+2))
      g <- matrix(0,(2*nj))
      w <- matrix(0,(2*nj))
      
      #Preencha a matriz 'matriz'
      for (i in 1:nj){
        t1 <- dados$mand[i]
        x1 <- which(times[]==t1)
        t2 <- dados$vist[i]
        x2 <- which(times[]==t2)
        
        matriz[((2*i)-1),1] <- 1
        matriz[ (2*i)   ,1] <- 1
        
        matriz[((2*i)-1),(1+x1)   ] <-  1
        matriz[((2*i)-1),(1+nt+x2)] <- -1
        
        matriz[(2*i),(1+x2)   ] <-  1
        matriz[(2*i),(1+nt+x1)] <- -1
        
        if (is.na(dados$Camp[i])) matriz[((2*i)-1),((2*nt)+2)] <- 1
        if (is.na(dados$Camp[i])) matriz[ (2*i)   ,((2*nt)+2)] <- 1
        
        g[((2*i)-1)] <- g1[i]
        g[ (2*i)   ] <- g2[i]
        
        w[((2*i)-1)] <- dados$peso[i]
        w[ (2*i)   ] <- dados$peso[i]
      }
      
      # Tentativa de ajuste do modelo com tratamento de erro
      bc <- tryCatch(
        { glm(g ~ matriz - 1, family = poisson, weights = w) },
        error = function(e) { "erro" }
      )
      
      # Verifica o conteúdo de 'bc'
      if (identical(bc, "erro")) {
        # caso que deu erro
        {
          dados2 <- subset(tab2, indice == n)
          #### escrevendo os dados.
          prev <- data.frame(
            id        = dados2$id,
            ano       = dados2$ano,
            rodada    = dados2$rodada,
            mandante  = dados2$mand,
            visitante = dados2$vist,
            pvm       = "IDT",
            pe        = "IDT",
            pvv       = "IDT",
            golman    = "IDT",
            golvis    = "IDT",
            lambda1    = 0,
            lambda2    = 0,
            resultado = "IDT"
          )
          if(plot){print(prev)}
          # Escrever no arquivo com cabeçalho apenas na primeira vez
          write_csv(
            prev,
            arquivo_saida,
            na = "NA",
            append = arquivo_existe # Escreve cabeçalho apenas se o arquivo não existe
          )
        }
      } else {
        #continuar sem erro
        { 
          times[nt+1] <- 'efeito casa'
          times[nt+2] <- 'intercepto'
          
          parametros <- data.frame(matrix(NA,(nt+2),3))
          parametros[,1] <- times
          parametros[(1:nt),2] <- coef(bc)[2:(nt+1)]
          parametros[(1:nt),3] <- coef(bc)[(nt+2):((2*nt)+1)]
          parametros[(nt+1),2] <- coef(bc)[((2*nt)+2)]
          parametros[(nt+2),2] <- coef(bc)[1]
          parametros[is.na(parametros)] <- 0
          parametros[(nt+1),3] <- NA
          parametros[(nt+2),3] <- NA
          colnames(parametros) <- c('Time','Ataque','Defesa')
          
          ###############################
          #contrução dos parametros
          {
            parametros2 <- matrix(0,(2*nt+2),1)
            parametros2[1,1]  <- parametros[nt+2,2]
            parametros2[(nt*2)+2,1] <- parametros[nt+1,2]
            
            for( j in 1:nt){
              parametros2[j+1,1]  <- parametros[j,2]
              parametros2[j+(nt+1),1]  <- parametros[j,3]
            }
            
            parametros2 <- as.matrix(as.numeric(parametros2))
          }
          ############################################################################
          #Previs?o para cada rodada do campeonato 
          dados2 <- subset(tab2, indice == n)
          
          man2 <- dados2$mand
          vis2 <- dados2$vist
          
          times2 <- times0
          
          nj2 <- dim(dados2)[1]
          nt2 <- length(times2)
          
          ##################################
          
          # construção da matriz
          
          matriz2 <- matrix(0,2,((2*nt)+2))
          
          t1 <- dados2$mand
          x1 <- which(times[]==t1)
          t2 <- dados2$vist
          x2 <- which(times[]==t2)
          
          matriz2[1,1        ] <-  1
          matriz2[1,(1+x1)]    <-  1
          matriz2[1,(1+nt+x2)] <- -1
          
          matriz2[2,1]         <-  1
          matriz2[2,(1+x2)]    <-  1
          matriz2[2,(1+nt+x1)] <- -1
          
          if (is.na(dados$Camp[i])) matriz2[1,((2*nt)+2)] <- 1
          if (is.na(dados$Camp[i])) matriz2[2,((2*nt)+2)] <- 1
          
          X <- matriz2%*%parametros2
          
          lambdax <- exp(X[1,1])
          lambday <- exp(X[2,1])
          
          # Holgate feito por meio de duas poisson
          
          pb1<-dpois(0:7, lambda = lambdax) 
          pb2<-dpois(0:7, lambda = lambday)
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
            lambda1    = lambdax,
            lambda2    = lambday
          )
          
          if(prev$pvm == prev$pe && prev$pe == prev$pvv){
            prev$resultado <- "INT"
          }else{
            prev$resultado <- ifelse(max(prev$pvm,prev$pe,prev$pvv) == prev$pvm, "vm", 
                                     ifelse(max(prev$pvm,prev$pe,prev$pvv) == prev$pvv, "vv", "em"))
          }
          
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
    } 
  }
  
  #fazer previsões fragmentadas considerando as rodadas anteriores
  {
    for (l in (nr+1):length(unique(tab_rod$indice))) {
      
      jogos <- tab_rod[tab_rod$indice==l,]
      jogos_ant <- tab_rod[(tab_rod$indice>=(l-nr) & tab_rod$indice<l),]
      
      dados <- jogos_ant
      
      man <- unique(dados$mand)
      vis <- unique(dados$vist)
      
      times <- times0
      
      g1 <- dados$plac_mand
      g2 <- dados$plac_vist
      
      nj <- nrow(dados)
      nt <- length(times)
      
      #Calculo da variavel peso
      {
        dados$peso <- 0
        
        for (i in 1:nrow(dados)) {
          dados$peso[i] = 1.4+(0.4*((dados$dia_num[i] -(dados$semestre[i] *730/2))-dados$dia_num[nj])/730)
          
          if(dados$peso[i]<0){dados$peso[i] <- 0}
        }
      }
      ##############################
      ### Método Chance I
      ##############################
      
      matriz <- matrix(0,(2*nj),((2*nt)+2))
      g <- matrix(0,(2*nj))
      w <- matrix(0,(2*nj))
      
      #Preencha a matriz 'matriz'
      for (i in 1:nj){
        t1 <- dados$mand[i]
        x1 <- which(times[]==t1)
        t2 <- dados$vist[i]
        x2 <- which(times[]==t2)
        
        matriz[((2*i)-1),1] <- 1
        matriz[ (2*i)   ,1] <- 1
        
        matriz[((2*i)-1),(1+x1)   ] <-  1
        matriz[((2*i)-1),(1+nt+x2)] <- -1
        
        matriz[(2*i),(1+x2)   ] <-  1
        matriz[(2*i),(1+nt+x1)] <- -1
        
        if (is.na(dados$Camp[i])) matriz[((2*i)-1),((2*nt)+2)] <- 1
        if (is.na(dados$Camp[i])) matriz[ (2*i)   ,((2*nt)+2)] <- 1
        
        g[((2*i)-1)] <- g1[i]
        g[ (2*i)   ] <- g2[i]
        
        w[((2*i)-1)] <- dados$peso[i]
        w[ (2*i)   ] <- dados$peso[i]
      }
      
      # Tentativa de ajuste do modelo com tratamento de erro
      bc <- tryCatch(
        { glm(g ~ matriz - 1, family = poisson, weights = w) },
        error = function(e) { "erro" }
      )
      
      # Verifica o conteúdo de 'bc'
      if (identical(bc, "erro")) {
        # caso que deu erro
        {
          dados2 <- jogos
          nj2 <- dim(dados2)[1]
          
          for (p in 1:nj2){
          #### escrevendo os dados.
          prev <- data.frame(
            id        = dados2$id[p],
            ano       = dados2$ano[p],
            rodada    = dados2$rodada[p],
            mandante  = dados2$mand[p],
            visitante = dados2$vist[p],
            pvm       = "IDT",
            pe        = "IDT",
            pvv       = "IDT",
            golman    = "IDT",
            golvis    = "IDT",
            lambda1    = 0,
            lambda2    = 0,
            resultado = "IDT"
          )
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
      } else {
        #continuar sem erro
        {
        times[nt+1] <- 'efeito casa'
        times[nt+2] <- 'intercepto'
        
        parametros <- data.frame(matrix(NA,(nt+2),3))
        parametros[,1] <- times
        parametros[(1:nt),2] <- coef(bc)[2:(nt+1)]
        parametros[(1:nt),3] <- coef(bc)[(nt+2):((2*nt)+1)]
        parametros[(nt+1),2] <- coef(bc)[((2*nt)+2)]
        parametros[(nt+2),2] <- coef(bc)[1]
        parametros[is.na(parametros)] <- 0
        parametros[(nt+1),3] <- NA
        parametros[(nt+2),3] <- NA
        colnames(parametros) <- c('Time','Ataque','Defesa')
        
        ###############################
        #contrução dos parametros
        #
        parametros2 <- matrix(0,(2*nt+2),1)
        parametros2[1,1]  <- parametros[nt+2,2]
        parametros2[(nt*2)+2,1] <- parametros[nt+1,2]
        
        for( j in 1:nt){
          parametros2[j+1,1]  <- parametros[j,2]
          parametros2[j+(nt+1),1]  <- parametros[j,3]
        }
        
        parametros2 <- as.matrix(as.numeric(parametros2))
        
        ############################################################################
        #Previs?o para cada rodada do campeonato 
        dados2 <- jogos
        
        man2 <- dados2$mand
        vis2 <- dados2$vist
        
        times2 <- times0
        
        nj2 <- dim(dados2)[1]
        nt2 <- length(times2)
        
        ##################################
        # construção da matriz
        
        matriz2 <- matrix(0,(2*nj2),((2*nt)+2))
        
        #Preencha a matriz 'matriz'
        for (i in 1:nj2){
          t1 <- dados2$mand[i]
          x1 <- which(times[]==t1)
          t2 <- dados2$vist[i]
          x2 <- which(times[]==t2)
          
          matriz2[((2*i)-1),1] <- 1
          matriz2[ (2*i)   ,1] <- 1
          
          matriz2[((2*i)-1),(1+x1)   ] <-  1
          matriz2[((2*i)-1),(1+nt+x2)] <- -1
          
          matriz2[(2*i),(1+x2)   ] <-  1
          matriz2[(2*i),(1+nt+x1)] <- -1
          
          if (is.na(dados2$Camp[i])) matriz2[((2*i)-1),((2*nt)+2)] <- 1
          if (is.na(dados2$Camp[i])) matriz2[ (2*i)   ,((2*nt)+2)] <- 1
        }
        
        for (p in 1:nj2) {
          
          matriz3 <- matriz2[c(2*p-1, 2*p),]
          
          X <- matriz3%*%parametros2
          
          lambdax <- exp(X[1,1])
          lambday <- exp(X[2,1])
          
          # Holgate feito por meio de duas poisson
          pb1<-dpois(0:7, lambda = lambdax) 
          pb2<-dpois(0:7, lambda = lambday)
          mh<-pb1%*%t(pb2)
          
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
            id        = dados2$id[p],
            ano       = dados2$ano[p],
            rodada    = dados2$rodada[p],
            mandante  = dados2$mand[p],
            visitante = dados2$vist[p],
            pvm       = VM,
            pe        = ET,
            pvv       = VV,
            golman    = (pplacar[1]-1),
            golvis    = (pplacar[2]-1),
            lambda1    = lambdax,
            lambda2    = lambday
          )
          
          if(prev$pvm == prev$pe && prev$pe == prev$pvv){
            prev$resultado <- "INT"
          }else{
            prev$resultado <- ifelse(max(prev$pvm,prev$pe,prev$pvv) == prev$pvm, "vm", 
                                     ifelse(max(prev$pvm,prev$pe,prev$pvv) == prev$pvv, "vv", "em"))
          }
          
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
      }
    } 
  }
  
  cat("Data frame", arquivo_saida, "de previsões criado e salvo na pasta.\n")
}

