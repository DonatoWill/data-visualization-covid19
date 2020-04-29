import streamlit as st
import pandas as pd
import numpy as np
import time as time
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
import plotly.io as pio
pio.renderers.default = "colab"
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

def main():
   
    # Lendo os arquivos csv
    df = pd.read_csv('covid-brasil.csv', delimiter=';')
    df_original = df
    
    # Montando lista com os estados e regiões
    states = np.append([''], df['estado'].unique())
    regions = np.append([''], df['regiao'].unique())
    
    # Título menu lateral
    st.sidebar.title("Filtros")

    # Seleção de dataset Brasil ou Mundo
    dataset_select = st.sidebar.selectbox("Escolha os dados: ", ('Mundo', 'Brasil'))

    if dataset_select == 'Brasil':
        st.header("Dados do COVID-19 no Brasil")
        # Input para seleção de tipo de gráfico 
        chart_mode = st.sidebar.selectbox("Escolha o tipo de gráfico: ", ('', 'Marcadores', 'Linha', 'Barras'))

        # Input para seleção da região 
        selected_region = st.sidebar.selectbox("Seleciona região: ", list(regions))

        # Verifica a região escolhida e filtra no dataframe
        if selected_region != '':
            states = np.append([''], df[df['regiao'] == selected_region]['estado'].unique())     
            df = df[df['regiao'] == selected_region]
            df_regiao = df[df['regiao'] == selected_region]
            st.write('Região selecionada: ', selected_region)
        elif selected_region == '':
            df_regiao = df

        # Input para seleção do estado 
        selected_state = st.sidebar.selectbox("Seleciona estado: ", list(states))

        # Verifica o estado selecionado e filtra no dataframe
        if selected_state != '':
            df = df[df['estado'] == selected_state]
            st.write('Estado selecionado: ', selected_state)

        # Agrupa os dados por data, estado e região em um novo dataframe
        df.data = pd.to_datetime(df.data)
        df_plot = pd.DataFrame({'casos' : df.groupby(['data', 'estado','regiao'])['casosAcumulados'].sum()}).reset_index()

        # Cria um input slidebar com o número de registros analisados
        slider = st.sidebar.slider("Registros analisados:", df_plot.index.start, df_plot.index.stop-1, df_plot.index.stop-1)

        fig = go.Figure()

        if chart_mode == '' or chart_mode == 'Linha':
            chart_mode = 'lines'
        elif chart_mode == 'Marcadores':
            chart_mode = 'markers'

        # Chama a função que retorna os traces de acordo com os filtros selecionados
        fig.add_traces(defineTraces(df_plot, slider, chart_mode, selected_region, selected_state))

        # Altera o layout
        fig.update_layout(
                autosize=False,
                width=900,
                height=500,
                margin=dict(
                    l=10,
                    r=10,
                    b=10,
                    t=10,
                    pad=4
                ),
                xaxis_title="Data",
                yaxis_title="Qtd. de Casos",
                xaxis_rangeslider_visible=True,
                showlegend=True
            )
        st.plotly_chart(fig)

        df_state = treatStateData(df, df_original)

        df_region = treatRegionData(df_regiao, df_original)
 
        fig_comapre = None
        pie_fig = None

        # Pie Chart
        if selected_region == '' and selected_state == '':
            pie_fig = px.pie(df_region, values='casos', names='regiao', labels={'casos': 'Casos', 'regiao': 'Região'})
            df_trace = df.groupby('data')['casosAcumulados', 'obitosAcumulados'].sum().reset_index()
            fig_comapre = go.Figure()
            trace = []
            trace.append(go.Scatter(x=df_trace.data, 
                    y=df_trace.casosAcumulados,
                    fill= 'tozeroy',
                    name= 'Casos'
                )
            )
            trace.append(go.Scatter(x=df_trace.data, 
                    y=df_trace.obitosAcumulados,
                    fill= 'tozeroy',
                    name= 'Mortes'
                )
            )
            fig_comapre.add_traces(trace)     

        elif selected_state == '':

            pie_fig = px.pie(df_state, values='casos', names='estado', labels={'casos': 'Casos', 'estado': 'Estado'})
            df_trace = df[(df['regiao'] == selected_region)]
            df_trace_group = df_trace.groupby('data')['casosAcumulados', 'obitosAcumulados'].sum().reset_index()
            fig_comapre = go.Figure()
            trace = []
            trace.append(go.Scatter(x=df_trace_group.data, 
                    y=df_trace_group.casosAcumulados,
                    fill= 'tozeroy',
                    name= 'Casos'
                )
            )
            trace.append(go.Scatter(x=df_trace_group.data, 
                    y=df_trace_group.obitosAcumulados,
                    fill= 'tozeroy',
                    name= 'Mortes'
                )
            )
            fig_comapre.add_traces(trace)         

        if pie_fig is not None:
            st.subheader('% de casos:')
            pie_fig.update_layout(
                    autosize=False,
                    width=900,
                    height=500,
                    margin=dict(
                        l=10,
                        r=10,
                        b=10,
                        t=10,
                        pad=4
                    ),
                    xaxis_rangeslider_visible=True,
                    showlegend=True
                )
            st.plotly_chart(pie_fig)

        if fig_comapre is not None:
            st.subheader('Comparação casos x mortes: ')
            fig_comapre.update_layout(
                            xaxis_rangeslider_visible=True,
                            width=900,
                            height=500,
                            margin=dict(
                                    l=10,
                                    r=10,
                                    b=10,
                                    t=10,
                                    pad=4
                                ))

            st.plotly_chart(fig_comapre)


        st.subheader('Dados por Estado')
        st.table(df_state.sort_values(by=['casos'], ascending=False))
        st.subheader('Dados por Região')
        st.table(df_region.sort_values(by=['casos'], ascending=False))

        
    
    elif dataset_select == 'Mundo':

        st.header("Dados do COVID-19 no Mundo")
        df_world = treatDataFrame()

        # Agrupando por data e região
        df_group = df_world.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

        # Agrupando por data
        df_group_date = df_world.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

        # Soma os tipos por data
        df_group_date_melt = df_group_date.melt(id_vars='Date', value_vars=['Recovered', 'Deaths', 'Active'], var_name='Case', value_name='Count')
        df_group_date_melt.loc[df_group_date_melt['Case'] == 'Recovered', 'Case'] = 'Recuperados'
        df_group_date_melt.loc[df_group_date_melt['Case'] == 'Deaths', 'Case'] = 'Mortes'
        df_group_date_melt.loc[df_group_date_melt['Case'] == 'Active', 'Case'] = 'Ativos'

        st.subheader('Dados por País: ')
        st.write("*Use o slider abaixo para visualizar a evolução.")

        fig2 = px.choropleth(df_group,
                    locations='Country/Region',
                    locationmode='country names',
                    color=np.log(df_group['Confirmed']),
                    hover_name='Country/Region',
                    hover_data=['Confirmed', 'Deaths'],
                    animation_frame=df_group['Date'].dt.strftime('%d-%m-%Y'),
                    color_continuous_scale= px.colors.sequential.Reds,
                    labels={'Confirmed': 'Confirmados', 'animation_frame': 'Data', 'Deaths': 'Mortes', 'Country/Region': 'País'}                 
                    )

        fig2.update_layout(autosize=False, 
                        width=1000, 
                        height=800,
                        geo = dict(
                                    showframe = True,
                                    showcoastlines = True,
                                    projection = dict(type = 'orthographic'),
                                    bgcolor= "rgba(255,255,255,1)",
                                    lakecolor="rgba(135,206,250,1)",
                                    oceancolor = "rgba(135,206,250,0.5)",
                                    rivercolor = "rgba(135,206,250,0.5)",
                                    landcolor = "rgba(255,255,255,1)",
                                    countrycolor= '#ff0000',
                                    coastlinecolor= '#ff0000',
                                    framecolor= '#ff0000',
                                    showocean=True,
                                    showsubunits=None
                                ),
                            hoverlabel = dict(),
                            margin=dict(
                                l=10,
                                r=10,
                                b=10,
                                t=10,
                                pad=4
                            )
                        )

        st.plotly_chart(fig2)

        st.subheader('Gráfico comparativo de casos (Recuperados, Mortes e Ativos): ')
        fig = px.area(df_group_date_melt,
                    x="Date",
                    y="Count",
                    color="Case",
                    width=1000,
                    height=600,
                    labels={'Case': 'Caso', 'Date': 'Data', 'Count': 'Qtd.'}  
                )

        fig.update_layout(xaxis_rangeslider_visible=True, margin=dict(
                                l=10,
                                r=10,
                                b=10,
                                t=10,
                                pad=4
                            ))

        st.plotly_chart(fig)


def treatRegionData(df_regiao, df_original):
        # Trata dados Casos por região
        df_region = pd.DataFrame( { 'casos': df_regiao.groupby(['regiao'])['casosNovos'].sum(), 
                                'obtos': df_regiao.groupby(['regiao'])['obitosNovos'].sum(),
                                'porcentagem': (df_regiao.groupby(['regiao'])['casosNovos'].sum() * 100 / df_original['casosNovos'].sum())}).reset_index()
        df_region['letalidade'] = np.round((df_region['obtos'] / df_region['casos']) * 100, 2)
        df_region['letalidade'] = df_region['letalidade'].apply(lambda x: str(x) + '%')
        df_region['porcentagem'] = np.round(df_region['porcentagem'], 2)
        df_region['porcentagem'] = df_region['porcentagem'].apply(lambda x: str(x) + '%')

        return df_region
    

def treatStateData(df, df_original):
        # Trata dados Casos por estado
        df_estado = pd.DataFrame( { 'casos': df.groupby(['estado'])['casosNovos'].sum(),
                                    'obtos': df.groupby(['estado'])['obitosNovos'].sum(),
                                    'porcentagem': (df.groupby(['estado'])['casosNovos'].sum() * 100 / df_original['casosNovos'].sum())}).reset_index()

        # Letalidade por estado
        df_estado['letalidade'] = np.round((df_estado['obtos'] / df_estado['casos']) * 100, 2)
        df_estado['letalidade'] = df_estado['letalidade'].apply(lambda x: str(x) + '%')
        df_estado['porcentagem'] = np.round(df_estado['porcentagem'], 2)
        df_estado['porcentagem'] = df_estado['porcentagem'].apply(lambda x: str(x) + '%')

        return df_estado

def treatDataFrame():

        df_world = pd.read_csv('covid_19_clean_complete.csv', delimiter=',', parse_dates=['Date'])
        df_world['Active'] = df_world['Confirmed'] - df_world['Deaths'] - df_world['Recovered']
        # Trocando Mainland China por China
        df_world['Country/Region'] = df_world['Country/Region'].replace('Mainland China', 'China')
        # Preenchendo missing values
        df_world[['Province/State']] = df_world[['Province/State']].fillna('')

        return df_world


def defineTraces(df, slider, mode, region, state):
    trace = []
    if region != '' and state == '':
        for state in df['estado'].unique():
            df_trace = df[(df['estado'] == state) & (df['data'] <= df.loc[slider].data)]
            trace.append(go.Scatter(x=df_trace.data, y=df_trace.casos,
                        mode= mode,
                        name=state
                    )
            )
        # Linha geral da região
        df_trace = df.groupby(['data', 'regiao'])['casos'].sum().reset_index()
        
        trace.append(go.Scatter(x=df_trace.data, y=df_trace.casos,
                        mode= mode,
                        name='Total '+region
                    )
            )
    elif region == '' and state == '':
        df_regiao = df
        df_trace = df[df['data'] <= df.loc[slider].data]
        df_plot_all = df_trace.groupby('data')['casos'].sum()
        df_plot_all = pd.DataFrame({'data': df_plot_all.index, 'casos' : df_plot_all.values})     
        trace.append(go.Scatter(x=df_plot_all.data, y=df_plot_all.casos,
                        mode= mode,
                        name='Brasil',
                        fill='tozeroy'
                    )
        )
    elif region != '' and state != '':
        df_trace = df[df['data'] <= df.loc[slider].data]
        trace.append(go.Scatter(x=df_trace.data, y=df_trace.casos,
                        mode= mode,
                        name= state
                    )
        )
    else:
        df_trace = df.groupby(['data', 'estado'])['casos'].sum().reset_index()
        trace.append(go.Scatter(x=df_trace.data, y=df_trace.casos,
                        mode= mode,
                        name=state
                    )
            )
    return trace


if __name__ == "__main__":
    main()