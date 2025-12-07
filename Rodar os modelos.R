library(tidyverse)

setwd("~/Documents/Dissertação/scripts")

br <- read_csv("bancos/br.csv", 
               col_types = cols(ID = col_number(),
                                rodata = col_number(), 
                                data = col_date(format = "%d/%m/%Y"), 
                                ano = col_number(),
                                hora = col_time(format = "%H:%M"), 
                                mandante_Placar = col_number(),
                                visitante_Placar = col_number()
               )
)
classicos <- read_csv("bancos/classicos.csv")
attach(br)

# carregando funções
{
  ### preparação de dados
  
  source("Funções dos modelos/prep_dados.r")
  # parametros: id,dia,ano,rodada,mand,vist,plac_mand,plac_vist
  
  ### ufmg com simulação uniforme
  {
    source("Funções dos modelos/mod.ufmg.comsimul.geral.r")
    # parametros: tab,p,complemento do nome do arquivo, número de simulações, plotar TouF
    source("Funções dos modelos/mod.ufmg.comsimul.ano.r")
    # parametros: tab,p,complemento do nome do arquivo, número de simulações, plotar TouF
    source("Funções dos modelos/mod.ufmg.comsimul.janela.r")
    # parametros: tab,p,complemento do nome do arquivo, número de simulações, plotar TouF
    source("Funções dos modelos/mod.ufmg.comsimul.rodada.r")
    # parametros: tab,p,complemento do nome do arquivo, número de simulações, número de rodadas, plotar TouF
  }
  
  ### ufmg sem simulação
  {
    source("Funções dos modelos/mod.ufmg.semsimul.geral.r")
    # parametros: tab,p,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.ufmg.semsimul.ano.r")
    # parametros: tab,p,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.ufmg.semsimul.janela.r")
    # parametros: tab,p,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.ufmg.semsimul.rodada.r")
    # parametros: tab,p,complemento do nome do arquivo, número de rodadas, plotar TouF
  }
  
  ### ufmg com adição de erro normal
  {
    source("Funções dos modelos/mod.ufmg.simulnormal.geral.r")
    # parametros: tab,p,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.ufmg.simulnormal.ano.r")
    # parametros: tab,p,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.ufmg.simulnormal.janela.r")
    # parametros: tab,p,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.ufmg.simulnormal.rodada.r")
    # parametros: tab,p,complemento do nome do arquivo, número de rodadas, plotar TouF
  }
  
  ### modelos controle
  {
    source("Funções dos modelos/mod.controle.mand.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.controle.emp.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.controle.vist.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.controle.poisson.r")
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.controle.uniforme.r")
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.controle.implicito.media.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.controle.media.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
  }
  
  ### modelos Arruda
  {
    source("Funções dos modelos/mod.arr.SD0.geral.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.SD0.ano.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.SD0.janela.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.SD0.rodada.r") 
    # parametros: tab,complemento do nome do arquivo, nº de rodadas a considerar, plotar TouF
    
    source("Funções dos modelos/mod.arr.SD1.geral.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.SD1.ano.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.SD1.janela.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.SD1.rodada.r") 
    # parametros: tab,complemento do nome do arquivo, nº de rodadas a considerar, plotar TouF
    
    source("Funções dos modelos/mod.arr.chance1.geral.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.chance1.ano.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.chance1.janela.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.chance1.rodada.r") 
    # parametros: tab,complemento do nome do arquivo, nº de rodadas a considerar, plotar TouF
    
    source("Funções dos modelos/mod.arr.chance2.geral.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.chance2.ano.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.chance2.janela.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.chance2.rodada.r") 
    # parametros: tab,complemento do nome do arquivo, nº de rodadas a considerar, plotar TouF
  }
  
  ### metricas
  {
    source("Funções dos modelos/met.taxa.acerto.r")
    # parametros: previsto, observado, retornar_comparacao = TouF
    source("Funções dos modelos/met.taxa.func.r")
    # parametros: previsto
    source("Funções dos modelos/met.epmp.r")
    # parametros: previsto, observado, pvm, pem, pvv, retornar_comparacao = TouF
    source("Funções dos modelos/met.definetti.r")
    # parametros: pvm, pem, pvv, resul, retornar_componentes = TouF
    source("Funções dos modelos/met.definetti.detalhada.r")
    # parametros: placar_man, placar_vis, prev_golman, prev_golvis, retornar_componentes = TouF
    source("Funções dos modelos/met.matriz.confusao.r")
    # parametros: previsto, observado, retornar_comparacao = TouF
    source("Funções dos modelos/met.vero.r")
    # parametros: placar_man, placar_vis, lambda1, lambda2, retornar_componentes = TouF
  }
}


prep_dados(ID,data,ano,rodata,mandante,visitante,mandante_Placar,visitante_Placar,classicos)

tab <- subset(tab, ano %in% c(2019))

setwd("~/Documents/Dissertação/scripts/teste2")
# rodar modelos
{
  #####
  mod.controle.mand(tab,, F)
  mod.controle.emp(tab,, F)
  mod.controle.vist(tab,, F)
  mod.controle.poisson(tab,, F)
  mod.controle.uniforme(tab,, F)
  mod.controle.implicito.media(tab,, F)
  mod.controle.media(tab,, F)
  #####
  mod.ufmg.comsimul.geral(tab,5,,10000,F)
  mod.ufmg.comsimul.ano(tab,5,,10000, F)
  mod.ufmg.comsimul.janela(tab,5,,10000, F)
  mod.ufmg.comsimul.rodada(tab,5,,10000,2, F)
  #####
  mod.ufmg.semsimul.geral(tab,5,, F)
  mod.ufmg.semsimul.ano(tab,5,, F)
  mod.ufmg.semsimul.janela(tab,5,, F)
  mod.ufmg.semsimul.rodada(tab,5,,2, F)
  #####
  mod.ufmg.simulnormal.geral(tab,5,, F)
  mod.ufmg.simulnormal.ano(tab,5,, F)
  mod.ufmg.simulnormal.janela(tab,5,, F)
  mod.ufmg.simulnormal.rodada(tab,5,,2, F)
  #####
  mod.arr.SD0.geral(tab,, F)
  mod.arr.SD0.ano(tab,, F)
  mod.arr.SD0.janela(tab,, F)
  mod.arr.SD0.rodada(tab,, 2, F)
  #####
  mod.arr.SD1.geral(tab,, F)
  mod.arr.SD1.ano(tab,, F)
  mod.arr.SD1.janela(tab,, F)
  mod.arr.SD1.rodada(tab,, 2, F)
  #####
  mod.arr.chance1.geral(tab,, F)
  mod.arr.chance1.ano(tab,, F)
  mod.arr.chance1.janela(tab,, F)
  mod.arr.chance1.rodada(tab,, 2, F)
  #####
  mod.arr.chance2.geral(tab,, F)
  mod.arr.chance2.ano(tab,, F)
  mod.arr.chance2.janela(tab,, F)
  mod.arr.chance2.rodada(tab,, 2, F)
  
}
