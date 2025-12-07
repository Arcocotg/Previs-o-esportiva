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
  }
}


prep_dados(ID,data,ano,rodata,mandante,visitante,mandante_Placar,visitante_Placar,classicos)

tab <- subset(tab, ano %in% c(2018,2019))

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

# ler tabelas dos modelos 
{ 
  #####
  prev.mod.controle.mand <- read_csv("prev.mod.controle.mand.csv",
                                     col_types = cols(pvm = col_number(), 
                                                      pe = col_number(), 
                                                      pvv = col_number()),
                                     col_names = T)
  prev.mod.controle.mand <- arrange(prev.mod.controle.mand, id)
  
  prev.mod.controle.vist <- read_csv("prev.mod.controle.vist.csv",
                                     col_types = cols(pvm = col_number(), 
                                                      pe = col_number(), 
                                                      pvv = col_number()),
                                     col_names = T)
  prev.mod.controle.vist <- arrange(prev.mod.controle.vist, id)
  
  prev.mod.controle.emp <- read_csv("prev.mod.controle.emp.csv",
                                     col_types = cols(pvm = col_number(), 
                                                      pe = col_number(), 
                                                      pvv = col_number()),
                                     col_names = T)
  prev.mod.controle.emp <- arrange(prev.mod.controle.emp, id)

  prev.mod.controle.poisson <- read_csv("prev.mod.controle.poisson.csv",
                                     col_types = cols(golman = col_number(), 
                                                      golvis = col_number()),
                                     col_names = T)
  prev.mod.controle.poisson <- arrange(prev.mod.controle.poisson, id)

  prev.mod.controle.uniforme <- read_csv("prev.mod.controle.uniforme.csv",
                                         col_types = cols(pvm = col_number(), 
                                                          pe = col_number(), 
                                                          pvv = col_number()),
                                         col_names = T)
  prev.mod.controle.uniforme <- arrange(prev.mod.controle.uniforme, id)

  prev.mod.controle.implicito.media <- read_csv("prev.mod.controle.implicito.media.csv",
                                                col_types = cols(pvm = col_number(), 
                                                                 pe = col_number(), 
                                                                 pvv = col_number(),
                                                                 golman = col_number(), 
                                                                 golvis = col_number()),
                                                col_names = T)
  prev.mod.controle.implicito.media <- arrange(prev.mod.controle.implicito.media, id)
  
  prev.mod.controle.media <- read_csv("prev.mod.controle.media.csv",
                                      col_types = cols(golman = col_number(), 
                                                       golvis = col_number()),
                                      col_names = T)
  prev.mod.controle.media <- arrange(prev.mod.controle.media, id)
  
  #####
  prev.mod.ufmg.comsimul.geral <- read_csv("prev.mod.ufmg.comsimul.geral.csv",
                                           col_types = cols(pvm = col_number(), 
                                                            pe = col_number(), 
                                                            pvv = col_number()),
                                           col_names = T)
  prev.mod.ufmg.comsimul.geral <- arrange(prev.mod.ufmg.comsimul.geral, id)
  
  prev.mod.ufmg.comsimul.ano <- read_csv("prev.mod.ufmg.comsimul.ano.csv",
                                         col_types = cols(pvm = col_number(), 
                                                          pe = col_number(), 
                                                          pvv = col_number()),
                                         col_names = T)
  prev.mod.ufmg.comsimul.ano <- arrange(prev.mod.ufmg.comsimul.ano, id)
  
  prev.mod.ufmg.comsimul.janela <- read_csv("prev.mod.ufmg.comsimul.janela.csv",
                                            col_types = cols(pvm = col_number(), 
                                                             pe = col_number(), 
                                                             pvv = col_number()),
                                            col_names = T)
  prev.mod.ufmg.comsimul.janela <- arrange(prev.mod.ufmg.comsimul.janela, id)
  
  prev.mod.ufmg.comsimul.rodada <- read_csv("prev.mod.ufmg.comsimul.rodada.csv",
                                            col_types = cols(pvm = col_number(), 
                                                             pe = col_number(), 
                                                             pvv = col_number()),
                                            col_names = T)
  prev.mod.ufmg.comsimul.rodada <- arrange(prev.mod.ufmg.comsimul.rodada, id)
  
  #####
  prev.mod.ufmg.semsimul.geral <- read_csv("prev.mod.ufmg.semsimul.geral.csv",
                                           col_types = cols(pvm = col_number(), 
                                                            pe = col_number(), 
                                                            pvv = col_number()),
                                           col_names = T)
  prev.mod.ufmg.semsimul.geral <- arrange(prev.mod.ufmg.semsimul.geral, id)
  
  prev.mod.ufmg.semsimul.ano <- read_csv("prev.mod.ufmg.semsimul.ano.csv",
                                         col_types = cols(pvm = col_number(), 
                                                          pe = col_number(), 
                                                          pvv = col_number()),
                                         col_names = T)
  prev.mod.ufmg.semsimul.ano <- arrange(prev.mod.ufmg.semsimul.ano, id)
 
  prev.mod.ufmg.semsimul.janela <- read_csv("prev.mod.ufmg.semsimul.janela.csv",
                                            col_types = cols(pvm = col_number(), 
                                                             pe = col_number(), 
                                                             pvv = col_number()),
                                            col_names = T)
  prev.mod.ufmg.semsimul.janela <- arrange(prev.mod.ufmg.semsimul.janela, id)
  
  prev.mod.ufmg.semsimul.rodada <- read_csv("prev.mod.ufmg.semsimul.rodada.csv",
                                            col_types = cols(pvm = col_number(), 
                                                             pe = col_number(), 
                                                             pvv = col_number()),
                                            col_names = T)
  prev.mod.ufmg.semsimul.rodada <- arrange(prev.mod.ufmg.semsimul.rodada, id)
  
  #####
  prev.mod.ufmg.simulnormal.geral <- read_csv("prev.mod.ufmg.simulnormal.geral.csv",
                                              col_types = cols(pvm = col_number(), 
                                                               pe = col_number(), 
                                                               pvv = col_number()),
                                              col_names = T)
  prev.mod.ufmg.simulnormal.geral <- arrange(prev.mod.ufmg.simulnormal.geral, id)
  
  prev.mod.ufmg.simulnormal.ano <- read_csv("prev.mod.ufmg.simulnormal.ano.csv",
                                            col_types = cols(pvm = col_number(), 
                                                             pe = col_number(), 
                                                             pvv = col_number()),
                                            col_names = T)
  prev.mod.ufmg.simulnormal.ano <- arrange(prev.mod.ufmg.simulnormal.ano, id)

  prev.mod.ufmg.simulnormal.janela <- read_csv("prev.mod.ufmg.simulnormal.janela.csv",
                                               col_types = cols(pvm = col_number(), 
                                                                pe = col_number(), 
                                                                pvv = col_number()),
                                               col_names = T)
  prev.mod.ufmg.simulnormal.janela <- arrange(prev.mod.ufmg.simulnormal.janela, id)
  
  prev.mod.ufmg.simulnormal.rodada <- read_csv("prev.mod.ufmg.simulnormal.rodada.csv",
                                               col_types = cols(pvm = col_number(), 
                                                                pe = col_number(), 
                                                                pvv = col_number()),
                                               col_names = T)
  prev.mod.ufmg.simulnormal.rodada <- arrange(prev.mod.ufmg.simulnormal.rodada, id)
  
  #####
  prev.mod.arr.SD0.geral <- read_csv("prev.mod.arr.SD0.geral.csv",
                                     col_types = cols(pvm = col_number(), 
                                                      pe = col_number(), 
                                                      pvv = col_number(),
                                                      golman = col_number(), 
                                                      golvis = col_number()),
                                     col_names = T)
  prev.mod.arr.SD0.geral <- arrange(prev.mod.arr.SD0.geral, id)
  
  prev.mod.arr.SD0.ano <- read_csv("prev.mod.arr.SD0.ano.csv",
                                   col_types = cols(pvm = col_number(), 
                                                    pe = col_number(), 
                                                    pvv = col_number(),
                                                    golman = col_number(), 
                                                    golvis = col_number()),
                                   col_names = T)
  prev.mod.arr.SD0.ano <- arrange(prev.mod.arr.SD0.ano, id)
  
  prev.mod.arr.SD0.janela <- read_csv("prev.mod.arr.SD0.janela.csv",
                                      col_types = cols(pvm = col_number(), 
                                                       pe = col_number(), 
                                                       pvv = col_number(),
                                                       golman = col_number(), 
                                                       golvis = col_number()),
                                      col_names = T)
  prev.mod.arr.SD0.janela <- arrange(prev.mod.arr.SD0.janela, id)
  
  prev.mod.arr.SD0.rodada <- read_csv("prev.mod.arr.SD0.rodada.csv",
                                      col_types = cols(pvm = col_number(), 
                                                       pe = col_number(), 
                                                       pvv = col_number(),
                                                       golman = col_number(), 
                                                       golvis = col_number()),
                                      col_names = T)
  prev.mod.arr.SD0.rodada <- arrange(prev.mod.arr.SD0.rodada, id)

  #####
  prev.mod.arr.SD1.geral <- read_csv("prev.mod.arr.SD1.geral.csv",
                                     col_types = cols(pvm = col_number(), 
                                                      pe = col_number(), 
                                                      pvv = col_number(),
                                                      golman = col_number(), 
                                                      golvis = col_number()),
                                     col_names = T)
  prev.mod.arr.SD1.geral <- arrange(prev.mod.arr.SD1.geral, id)
  
  prev.mod.arr.SD1.ano <- read_csv("prev.mod.arr.SD1.ano.csv",
                                   col_types = cols(pvm = col_number(), 
                                                    pe = col_number(), 
                                                    pvv = col_number(),
                                                    golman = col_number(), 
                                                    golvis = col_number()),
                                   col_names = T)
  prev.mod.arr.SD1.ano <- arrange(prev.mod.arr.SD1.ano, id)
  
  prev.mod.arr.SD1.janela <- read_csv("prev.mod.arr.SD1.janela.csv",
                                      col_types = cols(pvm = col_number(), 
                                                       pe = col_number(), 
                                                       pvv = col_number(),
                                                       golman = col_number(), 
                                                       golvis = col_number()),
                                      col_names = T)
  prev.mod.arr.SD1.janela <- arrange(prev.mod.arr.SD1.janela, id)
  
  prev.mod.arr.SD1.rodada <- read_csv("prev.mod.arr.SD1.rodada.csv",
                                      col_types = cols(pvm = col_number(), 
                                                       pe = col_number(), 
                                                       pvv = col_number(),
                                                       golman = col_number(), 
                                                       golvis = col_number()),
                                      col_names = T)
  prev.mod.arr.SD1.rodada <- arrange(prev.mod.arr.SD1.rodada, id)
  
  #####
  prev.mod.arr.chance1.geral <- read_csv("prev.mod.arr.chance1.geral.csv",
                                         col_types = cols(pvm = col_number(), 
                                                          pe = col_number(), 
                                                          pvv = col_number(),
                                                          golman = col_number(), 
                                                          golvis = col_number()),
                                         col_names = T)
  prev.mod.arr.chance1.geral <- arrange(prev.mod.arr.chance1.geral, id)
  
  prev.mod.arr.chance1.ano <- read_csv("prev.mod.arr.chance1.ano.csv",
                                       col_types = cols(pvm = col_number(), 
                                                        pe = col_number(), 
                                                        pvv = col_number(),
                                                        golman = col_number(), 
                                                        golvis = col_number()),
                                       col_names = T)
  prev.mod.arr.chance1.ano <- arrange(prev.mod.arr.chance1.ano, id)
  
  prev.mod.arr.chance1.janela <- read_csv("prev.mod.arr.chance1.janela.csv",
                                          col_types = cols(pvm = col_number(), 
                                                           pe = col_number(), 
                                                           pvv = col_number(),
                                                           golman = col_number(), 
                                                           golvis = col_number()),
                                          col_names = T)
  prev.mod.arr.chance1.janela <- arrange(prev.mod.arr.chance1.janela, id)
  
  prev.mod.arr.chance1.rodada <- read_csv("prev.mod.arr.chance1.rodada.csv",
                                          col_types = cols(pvm = col_number(), 
                                                           pe = col_number(), 
                                                           pvv = col_number(),
                                                           golman = col_number(), 
                                                           golvis = col_number()),
                                          col_names = T)
  prev.mod.arr.chance1.rodada <- arrange(prev.mod.arr.chance1.rodada, id)
  
  #####
  prev.mod.arr.chance2.geral <- read_csv("prev.mod.arr.chance2.geral.csv",
                                         col_types = cols(pvm = col_number(), 
                                                          pe = col_number(), 
                                                          pvv = col_number(),
                                                          golman = col_number(), 
                                                          golvis = col_number()),
                                         col_names = T)
  prev.mod.arr.chance2.geral <- arrange(prev.mod.arr.chance2.geral, id)
  
  prev.mod.arr.chance2.ano <- read_csv("prev.mod.arr.chance2.ano.csv",
                                       col_types = cols(pvm = col_number(), 
                                                        pe = col_number(), 
                                                        pvv = col_number(),
                                                        golman = col_number(), 
                                                        golvis = col_number()),
                                       col_names = T)
  prev.mod.arr.chance2.ano <- arrange(prev.mod.arr.chance2.ano, id)
  
  prev.mod.arr.chance2.janela <- read_csv("prev.mod.arr.chance2.janela.csv",
                                          col_types = cols(pvm = col_number(), 
                                                           pe = col_number(), 
                                                           pvv = col_number(),
                                                           golman = col_number(), 
                                                           golvis = col_number()),
                                          col_names = T)
  prev.mod.arr.chance2.janela <- arrange(prev.mod.arr.chance2.janela, id)
  
  prev.mod.arr.chance2.rodada <- read_csv("prev.mod.arr.chance2.rodada.csv",
                                          col_types = cols(pvm = col_number(), 
                                                           pe = col_number(), 
                                                           pvv = col_number(),
                                                           golman = col_number(), 
                                                           golvis = col_number()),
                                          col_names = T)
  prev.mod.arr.chance2.rodada <- arrange(prev.mod.arr.chance2.rodada, id)
}

{
  # Listar os modelos
{
  #modelos
  {modelos <- c(
    "ConImp",     # mod.controle.implicito.media
    "ConTeiMan",  # mod.controle.mand
    "ConTeiEmp",  # mod.controle.emp
    "ConTeiVis",  # mod.controle.vist
    "ConMed",     # mod.controle.media
    "ConPois",    # mod.controle.poisson
    "ConUnif",    # mod.controle.uniforme
    
    "Ch1Ano",     # mod.arr.chance1.ano
    "Ch1Ger",     # mod.arr.chance1.geral
    "Ch1Sem",     # mod.arr.chance1.janela
    "Ch1Rod",     # mod.arr.chance1.rodada
    
    "Ch2Ano",     # mod.arr.chance2.ano
    "Ch2Ger",     # mod.arr.chance2.geral
    "Ch2Sem",     # mod.arr.chance2.janela
    "Ch2Rod",     # mod.arr.chance2.rodada
    
    "Sd0Ano",     # mod.arr.SD0.ano
    "Sd0Ger",     # mod.arr.SD0.geral
    "Sd0Sem",     # mod.arr.SD0.janela
    "Sd0Rod",     # mod.arr.SD0.rodada
    
    "Sd1Ano",     # mod.arr.SD1.ano
    "Sd1Ger",     # mod.arr.SD1.geral
    "Sd1Sem",     # mod.arr.SD1.janela
    "Sd1Rod",     # mod.arr.SD1.rodada
    
    "UfmgAno",    # mod.ufmg.comsimul.ano
    "UfmgGer",    # mod.ufmg.comsimul.geral
    "UfmgSem",    # mod.ufmg.comsimul.janela
    "UfmgRod",    # mod.ufmg.comsimul.rodada
    
    "UfmgSsAno",  # mod.ufmg.semsimul.ano
    "UfmgSsGer",  # mod.ufmg.semsimul.geral
    "UfmgSsSem",  # mod.ufmg.semsimul.janela
    "UfmgSsRod",  # mod.ufmg.semsimul.rodada
    
    "UfmgSnAno",  # mod.ufmg.simulnormal.ano
    "UfmgSnGer",  # mod.ufmg.simulnormal.geral
    "UfmgSnSem",  # mod.ufmg.simulnormal.janela
    "UfmgSnRod"   # mod.ufmg.simulnormal.rodada
  )}
  
  #definetti
  {definetti <- c(
    met.definetti(prev.mod.controle.implicito.media$pvm, prev.mod.controle.implicito.media$pe, prev.mod.controle.implicito.media$pvv, tab$resul, F),
    met.definetti(prev.mod.controle.mand$pvm, prev.mod.controle.mand$pe, prev.mod.controle.mand$pvv, tab$resul, F),
    met.definetti(prev.mod.controle.emp$pvm, prev.mod.controle.emp$pe, prev.mod.controle.emp$pvv, tab$resul, F),
    met.definetti(prev.mod.controle.vist$pvm, prev.mod.controle.vist$pe, prev.mod.controle.vist$pvv, tab$resul, F),
    "Não se aplica",
    "Não se aplica",
    met.definetti(prev.mod.controle.uniforme$pvm, prev.mod.controle.uniforme$pe, prev.mod.controle.uniforme$pvv, tab$resul, F),
    
    met.definetti(prev.mod.arr.chance1.ano$pvm, prev.mod.arr.chance1.ano$pe, prev.mod.arr.chance1.ano$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.chance1.geral$pvm, prev.mod.arr.chance1.geral$pe, prev.mod.arr.chance1.geral$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.chance1.janela$pvm, prev.mod.arr.chance1.janela$pe, prev.mod.arr.chance1.janela$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.chance1.rodada$pvm, prev.mod.arr.chance1.rodada$pe, prev.mod.arr.chance1.rodada$pvv, tab$resul, F),
    
    met.definetti(prev.mod.arr.chance2.ano$pvm, prev.mod.arr.chance2.ano$pe, prev.mod.arr.chance2.ano$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.chance2.geral$pvm, prev.mod.arr.chance2.geral$pe, prev.mod.arr.chance2.geral$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.chance2.janela$pvm, prev.mod.arr.chance2.janela$pe, prev.mod.arr.chance2.janela$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.chance2.rodada$pvm, prev.mod.arr.chance2.rodada$pe, prev.mod.arr.chance2.rodada$pvv, tab$resul, F),
    
    met.definetti(prev.mod.arr.SD0.ano$pvm, prev.mod.arr.SD0.ano$pe, prev.mod.arr.SD0.ano$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.SD0.geral$pvm, prev.mod.arr.SD0.geral$pe, prev.mod.arr.SD0.geral$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.SD0.janela$pvm, prev.mod.arr.SD0.janela$pe, prev.mod.arr.SD0.janela$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.SD0.rodada$pvm, prev.mod.arr.SD0.rodada$pe, prev.mod.arr.SD0.rodada$pvv, tab$resul, F),
    
    met.definetti(prev.mod.arr.SD1.ano$pvm, prev.mod.arr.SD1.ano$pe, prev.mod.arr.SD1.ano$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.SD1.geral$pvm, prev.mod.arr.SD1.geral$pe, prev.mod.arr.SD1.geral$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.SD1.janela$pvm, prev.mod.arr.SD1.janela$pe, prev.mod.arr.SD1.janela$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.SD1.rodada$pvm, prev.mod.arr.SD1.rodada$pe, prev.mod.arr.SD1.rodada$pvv, tab$resul, F),
    
    met.definetti(prev.mod.ufmg.comsimul.ano$pvm, prev.mod.ufmg.comsimul.ano$pe, prev.mod.ufmg.comsimul.ano$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.comsimul.geral$pvm, prev.mod.ufmg.comsimul.geral$pe, prev.mod.ufmg.comsimul.geral$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.comsimul.janela$pvm, prev.mod.ufmg.comsimul.janela$pe, prev.mod.ufmg.comsimul.janela$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.comsimul.rodada$pvm, prev.mod.ufmg.comsimul.rodada$pe, prev.mod.ufmg.comsimul.rodada$pvv, tab$resul, F),
    
    met.definetti(prev.mod.ufmg.semsimul.ano$pvm, prev.mod.ufmg.semsimul.ano$pe, prev.mod.ufmg.semsimul.ano$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.semsimul.geral$pvm, prev.mod.ufmg.semsimul.geral$pe, prev.mod.ufmg.semsimul.geral$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.semsimul.janela$pvm, prev.mod.ufmg.semsimul.janela$pe, prev.mod.ufmg.semsimul.janela$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.semsimul.rodada$pvm, prev.mod.ufmg.semsimul.rodada$pe, prev.mod.ufmg.semsimul.rodada$pvv, tab$resul, F),
    
    met.definetti(prev.mod.ufmg.simulnormal.ano$pvm, prev.mod.ufmg.simulnormal.ano$pe, prev.mod.ufmg.simulnormal.ano$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.simulnormal.geral$pvm, prev.mod.ufmg.simulnormal.geral$pe, prev.mod.ufmg.simulnormal.geral$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.simulnormal.janela$pvm, prev.mod.ufmg.simulnormal.janela$pe, prev.mod.ufmg.simulnormal.janela$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.simulnormal.rodada$pvm, prev.mod.ufmg.simulnormal.rodada$pe, prev.mod.ufmg.simulnormal.rodada$pvv, tab$resul, F)
  )}
  
  # definetti detalhada
  {definetti.detalhada <- c(
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.controle.implicito.media$golman, prev.mod.controle.implicito.media$golvis, F),
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.controle.media$golman, prev.mod.controle.media$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.controle.poisson$golman, prev.mod.controle.poisson$golvis, F),
    "Não se aplica",
    
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance1.ano$golman, prev.mod.arr.chance1.ano$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance1.geral$golman, prev.mod.arr.chance1.geral$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance1.janela$golman, prev.mod.arr.chance1.janela$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance1.rodada$golman, prev.mod.arr.chance1.rodada$golvis, F),
    
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance2.ano$golman, prev.mod.arr.chance2.ano$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance2.geral$golman, prev.mod.arr.chance2.geral$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance2.janela$golman, prev.mod.arr.chance2.janela$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance2.rodada$golman, prev.mod.arr.chance2.rodada$golvis, F),
    
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD0.ano$golman, prev.mod.arr.SD0.ano$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD0.geral$golman, prev.mod.arr.SD0.geral$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD0.janela$golman, prev.mod.arr.SD0.janela$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD0.rodada$golman, prev.mod.arr.SD0.rodada$golvis, F),
    
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD1.ano$golman, prev.mod.arr.SD1.ano$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD1.geral$golman, prev.mod.arr.SD1.geral$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD1.janela$golman, prev.mod.arr.SD1.janela$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD1.rodada$golman, prev.mod.arr.SD1.rodada$golvis, F),
    
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    "Não se aplica"
  )}
  
  #taxa de acerto
  {taxa_acerto <- c(
    met.taxa.acerto(prev.mod.controle.implicito.media$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.controle.mand$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.controle.emp$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.controle.vist$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.controle.media$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.controle.poisson$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.controle.uniforme$resultado, tab$resul, FALSE),
    
    met.taxa.acerto(prev.mod.arr.chance1.ano$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.chance1.geral$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.chance1.janela$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.chance1.rodada$resultado, tab$resul, FALSE),
    
    met.taxa.acerto(prev.mod.arr.chance2.ano$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.chance2.geral$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.chance2.janela$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.chance2.rodada$resultado, tab$resul, FALSE),
    
    met.taxa.acerto(prev.mod.arr.SD0.ano$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.SD0.geral$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.SD0.janela$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.SD0.rodada$resultado, tab$resul, FALSE),
    
    met.taxa.acerto(prev.mod.arr.SD1.ano$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.SD1.geral$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.SD1.janela$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.SD1.rodada$resultado, tab$resul, FALSE),
    
    met.taxa.acerto(prev.mod.ufmg.comsimul.ano$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.comsimul.geral$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.comsimul.janela$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.comsimul.rodada$resultado, tab$resul, FALSE),
    
    met.taxa.acerto(prev.mod.ufmg.semsimul.ano$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.semsimul.geral$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.semsimul.janela$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.semsimul.rodada$resultado, tab$resul, FALSE),
    
    met.taxa.acerto(prev.mod.ufmg.simulnormal.ano$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.simulnormal.geral$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.simulnormal.janela$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.simulnormal.rodada$resultado, tab$resul, FALSE)
  )}
  
  #epmp
  {epmp <- c(
    met.epmp(prev.mod.controle.implicito.media$resultado, tab$resul, prev.mod.controle.implicito.media$pvm, prev.mod.controle.implicito.media$pe, prev.mod.controle.implicito.media$pvv, FALSE),
    met.epmp(prev.mod.controle.mand$resultado, tab$resul, prev.mod.controle.mand$pvm, prev.mod.controle.mand$pe, prev.mod.controle.mand$pvv, FALSE),
    met.epmp(prev.mod.controle.emp$resultado, tab$resul, prev.mod.controle.emp$pvm, prev.mod.controle.emp$pe, prev.mod.controle.emp$pvv, FALSE),
    met.epmp(prev.mod.controle.vist$resultado, tab$resul, prev.mod.controle.vist$pvm, prev.mod.controle.vist$pe, prev.mod.controle.vist$pvv, FALSE),
    "Não se aplica",
    "Não se aplica",
    met.epmp(prev.mod.controle.uniforme$resultado, tab$resul, prev.mod.controle.uniforme$pvm, prev.mod.controle.uniforme$pe, prev.mod.controle.uniforme$pvv, FALSE),
    
    met.epmp(prev.mod.arr.chance1.ano$resultado, tab$resul, prev.mod.arr.chance1.ano$pvm, prev.mod.arr.chance1.ano$pe, prev.mod.arr.chance1.ano$pvv, FALSE),
    met.epmp(prev.mod.arr.chance1.geral$resultado, tab$resul, prev.mod.arr.chance1.geral$pvm, prev.mod.arr.chance1.geral$pe, prev.mod.arr.chance1.geral$pvv, FALSE),
    met.epmp(prev.mod.arr.chance1.janela$resultado, tab$resul, prev.mod.arr.chance1.janela$pvm, prev.mod.arr.chance1.janela$pe, prev.mod.arr.chance1.janela$pvv, FALSE),
    met.epmp(prev.mod.arr.chance1.rodada$resultado, tab$resul, prev.mod.arr.chance1.rodada$pvm, prev.mod.arr.chance1.rodada$pe, prev.mod.arr.chance1.rodada$pvv, FALSE),
    
    met.epmp(prev.mod.arr.chance2.ano$resultado, tab$resul, prev.mod.arr.chance2.ano$pvm, prev.mod.arr.chance2.ano$pe, prev.mod.arr.chance2.ano$pvv, FALSE),
    met.epmp(prev.mod.arr.chance2.geral$resultado, tab$resul, prev.mod.arr.chance2.geral$pvm, prev.mod.arr.chance2.geral$pe, prev.mod.arr.chance2.geral$pvv, FALSE),
    met.epmp(prev.mod.arr.chance2.janela$resultado, tab$resul, prev.mod.arr.chance2.janela$pvm, prev.mod.arr.chance2.janela$pe, prev.mod.arr.chance2.janela$pvv, FALSE),
    met.epmp(prev.mod.arr.chance2.rodada$resultado, tab$resul, prev.mod.arr.chance2.rodada$pvm, prev.mod.arr.chance2.rodada$pe, prev.mod.arr.chance2.rodada$pvv, FALSE),
    
    met.epmp(prev.mod.arr.SD0.ano$resultado, tab$resul, prev.mod.arr.SD0.ano$pvm, prev.mod.arr.SD0.ano$pe, prev.mod.arr.SD0.ano$pvv, FALSE),
    met.epmp(prev.mod.arr.SD0.geral$resultado, tab$resul, prev.mod.arr.SD0.geral$pvm, prev.mod.arr.SD0.geral$pe, prev.mod.arr.SD0.geral$pvv, FALSE),
    met.epmp(prev.mod.arr.SD0.janela$resultado, tab$resul, prev.mod.arr.SD0.janela$pvm, prev.mod.arr.SD0.janela$pe, prev.mod.arr.SD0.janela$pvv, FALSE),
    met.epmp(prev.mod.arr.SD0.rodada$resultado, tab$resul, prev.mod.arr.SD0.rodada$pvm, prev.mod.arr.SD0.rodada$pe, prev.mod.arr.SD0.rodada$pvv, FALSE),
    
    met.epmp(prev.mod.arr.SD1.ano$resultado, tab$resul, prev.mod.arr.SD1.ano$pvm, prev.mod.arr.SD1.ano$pe, prev.mod.arr.SD1.ano$pvv, FALSE),
    met.epmp(prev.mod.arr.SD1.geral$resultado, tab$resul, prev.mod.arr.SD1.geral$pvm, prev.mod.arr.SD1.geral$pe, prev.mod.arr.SD1.geral$pvv, FALSE),
    met.epmp(prev.mod.arr.SD1.janela$resultado, tab$resul, prev.mod.arr.SD1.janela$pvm, prev.mod.arr.SD1.janela$pe, prev.mod.arr.SD1.janela$pvv, FALSE),
    met.epmp(prev.mod.arr.SD1.rodada$resultado, tab$resul, prev.mod.arr.SD1.rodada$pvm, prev.mod.arr.SD1.rodada$pe, prev.mod.arr.SD1.rodada$pvv, FALSE),
    
    met.epmp(prev.mod.ufmg.comsimul.ano$resultado, tab$resul, prev.mod.ufmg.comsimul.ano$pvm, prev.mod.ufmg.comsimul.ano$pe, prev.mod.ufmg.comsimul.ano$pvv, FALSE),
    met.epmp(prev.mod.ufmg.comsimul.geral$resultado, tab$resul, prev.mod.ufmg.comsimul.geral$pvm, prev.mod.ufmg.comsimul.geral$pe, prev.mod.ufmg.comsimul.geral$pvv, FALSE),
    met.epmp(prev.mod.ufmg.comsimul.janela$resultado, tab$resul, prev.mod.ufmg.comsimul.janela$pvm, prev.mod.ufmg.comsimul.janela$pe, prev.mod.ufmg.comsimul.janela$pvv, FALSE),
    met.epmp(prev.mod.ufmg.comsimul.rodada$resultado, tab$resul, prev.mod.ufmg.comsimul.rodada$pvm, prev.mod.ufmg.comsimul.rodada$pe, prev.mod.ufmg.comsimul.rodada$pvv, FALSE),
    
    met.epmp(prev.mod.ufmg.semsimul.ano$resultado, tab$resul, prev.mod.ufmg.semsimul.ano$pvm, prev.mod.ufmg.semsimul.ano$pe, prev.mod.ufmg.semsimul.ano$pvv, FALSE),
    met.epmp(prev.mod.ufmg.semsimul.geral$resultado, tab$resul, prev.mod.ufmg.semsimul.geral$pvm, prev.mod.ufmg.semsimul.geral$pe, prev.mod.ufmg.semsimul.geral$pvv, FALSE),
    met.epmp(prev.mod.ufmg.semsimul.janela$resultado, tab$resul, prev.mod.ufmg.semsimul.janela$pvm, prev.mod.ufmg.semsimul.janela$pe, prev.mod.ufmg.semsimul.janela$pvv, FALSE),
    met.epmp(prev.mod.ufmg.semsimul.rodada$resultado, tab$resul, prev.mod.ufmg.semsimul.rodada$pvm, prev.mod.ufmg.semsimul.rodada$pe, prev.mod.ufmg.semsimul.rodada$pvv, FALSE),
    
    met.epmp(prev.mod.ufmg.simulnormal.ano$resultado, tab$resul, prev.mod.ufmg.simulnormal.ano$pvm, prev.mod.ufmg.simulnormal.ano$pe, prev.mod.ufmg.simulnormal.ano$pvv, FALSE),
    met.epmp(prev.mod.ufmg.simulnormal.geral$resultado, tab$resul, prev.mod.ufmg.simulnormal.geral$pvm, prev.mod.ufmg.simulnormal.geral$pe, prev.mod.ufmg.simulnormal.geral$pvv, FALSE),
    met.epmp(prev.mod.ufmg.simulnormal.janela$resultado, tab$resul, prev.mod.ufmg.simulnormal.janela$pvm, prev.mod.ufmg.simulnormal.janela$pe, prev.mod.ufmg.simulnormal.janela$pvv, FALSE),
    met.epmp(prev.mod.ufmg.simulnormal.rodada$resultado, tab$resul, prev.mod.ufmg.simulnormal.rodada$pvm, prev.mod.ufmg.simulnormal.rodada$pe, prev.mod.ufmg.simulnormal.rodada$pvv, FALSE)
  )}
  
  #taxa de funcionamento
  {taxa_func <- c(
    met.taxa.func(prev.mod.controle.implicito.media$resultado),
    met.taxa.func(prev.mod.controle.mand$resultado),
    met.taxa.func(prev.mod.controle.emp$resultado),
    met.taxa.func(prev.mod.controle.vist$resultado),
    met.taxa.func(prev.mod.controle.media$resultado),
    met.taxa.func(prev.mod.controle.poisson$resultado),
    met.taxa.func(prev.mod.controle.uniforme$resultado),
    
    met.taxa.func(prev.mod.arr.chance1.ano$resultado),
    met.taxa.func(prev.mod.arr.chance1.geral$resultado),
    met.taxa.func(prev.mod.arr.chance1.janela$resultado),
    met.taxa.func(prev.mod.arr.chance1.rodada$resultado),
    
    met.taxa.func(prev.mod.arr.chance2.ano$resultado),
    met.taxa.func(prev.mod.arr.chance2.geral$resultado),
    met.taxa.func(prev.mod.arr.chance2.janela$resultado),
    met.taxa.func(prev.mod.arr.chance2.rodada$resultado),
    
    met.taxa.func(prev.mod.arr.SD0.ano$resultado),
    met.taxa.func(prev.mod.arr.SD0.geral$resultado),
    met.taxa.func(prev.mod.arr.SD0.janela$resultado),
    met.taxa.func(prev.mod.arr.SD0.rodada$resultado),
    
    met.taxa.func(prev.mod.arr.SD1.ano$resultado),
    met.taxa.func(prev.mod.arr.SD1.geral$resultado),
    met.taxa.func(prev.mod.arr.SD1.janela$resultado),
    met.taxa.func(prev.mod.arr.SD1.rodada$resultado),
    
    met.taxa.func(prev.mod.ufmg.comsimul.ano$resultado),
    met.taxa.func(prev.mod.ufmg.comsimul.geral$resultado),
    met.taxa.func(prev.mod.ufmg.comsimul.janela$resultado),
    met.taxa.func(prev.mod.ufmg.comsimul.rodada$resultado),
    
    met.taxa.func(prev.mod.ufmg.semsimul.ano$resultado),
    met.taxa.func(prev.mod.ufmg.semsimul.geral$resultado),
    met.taxa.func(prev.mod.ufmg.semsimul.janela$resultado),
    met.taxa.func(prev.mod.ufmg.semsimul.rodada$resultado),
    
    met.taxa.func(prev.mod.ufmg.simulnormal.ano$resultado),
    met.taxa.func(prev.mod.ufmg.simulnormal.geral$resultado),
    met.taxa.func(prev.mod.ufmg.simulnormal.janela$resultado),
    met.taxa.func(prev.mod.ufmg.simulnormal.rodada$resultado)
  )}
  
  #nível de complexidade
  {Complexidade <- c(
    2, # mod.controle.implicito.media
    1, # mod.controle.mand
    1, # mod.controle.emp
    1, # mod.controle.vist
    1, # mod.controle.media
    2, # mod.controle.poisson
    2, # mod.controle.uniforme
    
    4, # mod.arr.chance1.ano
    4, # mod.arr.chance1.geral
    4, # mod.arr.chance1.janela
    4, # mod.arr.chance1.rodada
    
    5, # mod.arr.chance2.ano
    5, # mod.arr.chance2.geral
    5, # mod.arr.chance2.janela
    5, # mod.arr.chance2.rodada
    
    4, # mod.arr.SD0.ano
    4, # mod.arr.SD0.geral
    4, # mod.arr.SD0.janela
    4, # mod.arr.SD0.rodada
    
    5, # mod.arr.SD1.ano
    5, # mod.arr.SD1.geral
    5, # mod.arr.SD1.janela
    5, # mod.arr.SD1.rodada
    
    3, # mod.ufmg.comsimul.ano
    3, # mod.ufmg.comsimul.geral
    3, # mod.ufmg.comsimul.janela
    3, # mod.ufmg.comsimul.rodada
    
    3, # mod.ufmg.semsimul.ano
    3, # mod.ufmg.semsimul.geral
    3, # mod.ufmg.semsimul.janela
    3, # mod.ufmg.semsimul.rodada
    
    4, # mod.ufmg.simulnormal.ano
    4, # mod.ufmg.simulnormal.geral
    4, # mod.ufmg.simulnormal.janela
    4 # mod.ufmg.simulnormal.rodada
  )}
  
  # Converter a lista em um data.frame
  resultados_df <- data.frame(
    Modelo = modelos,
    `Taxa de Acerto` = taxa_acerto,
    `Erro preditivo ponderado médio`= epmp,
    `Medida de Definetti`= definetti,
    `Medida de Definetti detalhada`= definetti.detalhada,
    `Taxa de Funcionamento` = taxa_func,
    `Nível de Complexidade` = Complexidade
  )
}
  # Exibir o resultado
  print(resultados_df)
  
  resultados_df <- resultados_df |>
    arrange(Taxa.de.Acerto)
  
  
  library(knitr)
  kable(resultados_df, format = "markdown", col.names = c("Modelo", 
                                                          "Taxa de Acerto", 
                                                          "Erro preditivo ponderado médio", 
                                                          "Medida de Definetti", 
                                                          "Medida de Definetti detalhada",
                                                          "Taxa de Funcionamento",
                                                          "Nível de Complexidade"))
}

# plotar grafico da TA

ggplot(resultados_df, aes(x = Taxa.de.Acerto, y = reorder(Modelo, Taxa.de.Acerto))) +
  geom_bar(stat = "identity", fill = "tomato") +
  geom_text(aes(label = round(Taxa.de.Acerto, 3)), hjust = -0.1, size = 3.8) +
  labs(
    x = "Taxa de Acerto",
    y = "Modelo",
    title = NULL
  ) +
  scale_x_continuous(
    breaks = seq(0, max(resultados_df$Taxa.de.Acerto), by = 0.05),
    expand = expansion(mult = c(0, 0.1))
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_line(color = "lightgray", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11),
    axis.text.y = element_text(
      angle = 0,
      hjust = 0,
      vjust = 0.5,
      lineheight = 2
    )
  )

# Exportar o gráfico (ajuste tamanho conforme quantidade de modelos)
ggsave("graf_taxa_acerto.png", plot = last_plot(), width = 13, height = 20, units = "cm", dpi = 300)



