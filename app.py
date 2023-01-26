
#--------------------------------------------------------------------------------------------------IMPORTAMOS LIBRERIAS---------------------------------------------------------------------------------------------

# librerias básicas
import os
import re
import requests
import pandas as pd
import numpy as np

# librerias machine learning para regresión lineal
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn import preprocessing

# mapas interactivos
import folium
from folium.plugins import FastMarkerCluster

# plotear gráficos y visualización
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import plot
import chart_studio.plotly as py
import plotly.io as pio

# streamlit
import streamlit as st
import streamlit.components.v1 as components
import google 
from PIL import Image
from pyngrok import ngrok
from IPython.display import display
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import urllib.request
import unicodedata
from unicodedata import name

# editor de texto
from wordcloud import WordCloud

#--------------------------------------------------------------------------------------------------PREPROCESAMIENTO----------------------------------------------------------------------------------------------------------
# leemos el dataframe que vamos a usar
listings = pd.read_csv('listings.csv.gz', compression = 'gzip')
listingsnulls = pd.read_csv('listings.csv.gz', compression = 'gzip')
security = pd.read_csv('crime.csv')

# añadrimos el preprocesamiento entero

# lista de columnas a eliminar que sobrepasan el 30% de valores nulos
columns_to_drop = ["neighbourhood_group_cleansed", "calendar_updated", "bathrooms", "license", "host_about", "host_neighbourhood", "neighborhood_overview", "neighbourhood"]
listings.drop(columns_to_drop, axis=1, inplace=True)

# creamos una función para rellenar los valores nulos por la media o la moda según el tipo de dato
def fillna(df, columns):
    """fillna rellena los valores nulos en un DataFrame con la media o moda dependiendo del tipo de columna.
    :param df: DataFrame que contiene los datos.
    :param columns: Lista con los nombres de las columnas que contienen
    :return: Devuelve dataframe con los valores nulos rellenados.
    """
    for column in columns:
        if(df[column].dtype == 'float64' or df[column].dtype == 'int64'):
            number = df[column].mean()
        else:
            number = df[column].mode()[0]
        df[column]= df[column].fillna(number)
    return df
listings = fillna(listings, listings.columns)

# creamos una función para eliminar los outliers
def outliers_repair (df):
    """outliers_repair se usa para identificar y reparar valores atípicos en un DataFrame dado.
    :param listings: DataFrame para el cual se deben reparar los valores atípicos.
    :returns: dataFrame sin valores atípicos.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

    dataset_wo_outliers = df.copy()
    dataset_wo_outliers = pd.DataFrame(df)
    for col in df.columns:
        if df[col].dtype == 'object':
            mode = df[col].mode()[0]
            dataset_wo_outliers.loc[outliers[col], col] = mode
        else:
            mean = df[col].mean()
            dataset_wo_outliers.loc[outliers[col], col] = mean
    return dataset_wo_outliers
listings = pd.DataFrame(outliers_repair(listings))

# hacemos una función para eliminar el simbolo del $ de las columnas necesarias
def remove_sign(df, column):
    """remove_sign se usa para eliminar un simbolo de en este caso un número
    :param listings: dataframe que estamos usando
    :param price: columna que queremos usar
    :return: dataframe con el texto eliminado de la columna
    """
    listings['price'] = listings['price'].apply(lambda x: x.replace("$", ""))    
    return listings
listings = remove_sign(listings, 'price')

# hacemos una función para eliminar las comas y convertimos el precio a float
def fix_price_column(df):
    """fix_price_column quita las comas de la columa que seleccionemos de nuestro dataframe y cambia los valores a float
    :param df: Dataframe sobre el que vamos a usar esta función
    :return: el valor corregido dentro de la columna asignada
    """
    listings['price'] = listings['price'].str.replace(',','')
    listings['price'] = listings['price'].astype(float)
    return listings
listings = fix_price_column(listings)

# hacemos una función para eliminar una longitud que da problemas a la hora de representar los mapas
def drop_rows(df, column, value):
    """drop rows elimina el valor que queremos de una fila de una columna
    :param df: dataframe que vamos a usar
    :param column: columna que tendrá los valores de las filas que queremos borrar
    :param value: valor que queremos borrar
    :return: devuelve el dataframe 
    """
    df = df[df[column] != value]
    return df
listings = drop_rows(listings, "longitude",  -122.43018616682234)

# creamos una función lambda que nos permita agrupar la distribución en los precios normales
listings['price'] = listings['price'].apply(lambda x: int(x/1000) if x >= 10000 else (int(x/100) if x >= 1000 else x))

# creamos una función para eliminar dentro de la columna precio los outliers
def remove_outliers(df, column):
    """esta función elimina valores outliers dentro de una columna de nuestro def
        :param df: llama al dataframe
        :param column: columna a eliminar los outliers
        :return: retorna el df sin outliers en nla columna seleccionada
    """
    df_clean = df[df[column].notnull()]
    mean = df[column].mean()
    std = df[column].std()
    outliers = df[(df[column] < mean - 2*std) | (df[column] > mean + 2*std)].index # std*2 es el doble de la desviación estandar que sirve para definir un rango de valores "normales" para una variable
    df.drop(outliers, inplace=True)
    return df
listings = remove_outliers(listings, 'price')

# agrupamos los barrios en sus distritos para una mejor lectura y creamos un nueva columna dentro del dataframe
listings['districts'] = ''
listings.loc[listings['neighbourhood_cleansed'].isin(['Pacific Heights', 'Nob Hill', 'Presidio Heights', 'Russian Hill']), 'districts'] = 'Central/downtown'
listings.loc[listings['neighbourhood_cleansed'].isin(['Western Addition', 'Haight Ashbury', 'Glen Park', 'Downtown/Civic Center', 'Financial District', 'Marina', 'North Beach', 'Chinatown']), 'districts'] = 'Upper Market and beyond (south central)'
listings.loc[listings['neighbourhood_cleansed'].isin(['Bernal Heights', 'Bayview', 'Excelsior', 'Outer Mission', 'Inner Sunset', 'Visitacion Valley', 'Crocker Amazon', 'Ocean View', 'Parkside']), 'districts'] = 'Bernal Heights/Bayview and beyond (southeast)'
listings.loc[listings['neighbourhood_cleansed'].isin(['Mission', 'Castro/Upper Market', 'Potrero Hill', 'South of Market', 'Outer Sunset']), 'districts'] = 'Upper Market and beyond (south central)'
listings.loc[listings['neighbourhood_cleansed'].isin(['Outer Richmond', 'Inner Richmond', 'West of Twin Peaks', 'Noe Valley', 'Twin Peaks', 'Golden Gate Park', 'Presidio', 'Lakeshore', 'Diamond Heights', 'Seacliff']), 'districts'] = 'Richmond'
listings.loc[listings['neighbourhood_cleansed'].isin(['Inner Richmond', 'Inner Sunset', 'Parkside', 'Outer Sunset']), 'districts'] = 'Sunset'


#------------------------------------------------------------------------------------COMENZAMOS LA APP -----------------------------------------------------------------------------------------------
st.set_page_config(page_title='SAN FRANCISCO' , layout='centered' , page_icon="✈️")
image = Image.open('C:\\Users\\rober\\Documents\\Rober\\Bootcamp\\Modulo 2\\20-Trabajo Módulo 2\\puentewide.png')
st.image(image, width=800)
st.write('Puente de San Francisco: Imagen creada con DALL-E-2')
st.title('EDA AIRBNB: SAN FRANCISCO')


# creamos una side bar y añadimos una barra con una pagina que nos muestra el tiempo en San Francisco
st.sidebar.title("El tiempo en San Francisco")
st.sidebar.write(f'<iframe src="https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629" width="" height="600" style="overflow:auto"></iframe>', unsafe_allow_html=True)
url = "https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629"
st.sidebar.markdown("[Accuweather](%s)" % url)


# creamos las pestañas que van a dividir nuestra app
tabs = st.tabs(['¿QUE VER EN SAN FRANCISCO?','INSIDE AIRBNB','PREPROCESAMIENTO', 'EDA', 'CONCLUSIONES', 'REGRESIÓN LINEAL'])


#------------------------------------------------------------------------PRIMERA PESTAÑA: ¿QUE VER EN SAN FRANCISCO?-------------------------------------------------------------------------------
tab_plots = tabs[0]

# añadimos una página para ver los mejores sitios de San Francisco
with tab_plots:
    st.header('¿Que ver en San Francisco?')
    st.markdown("""San Francisco es uno de los destinos turísticos más populares de los Estados Unidos.
                La ciudad ofrece una variedad de actividades para disfrutar como recorrer la famosa Golden Gate Bridge, 
                explorar la isla de Alcatraz, visitar el Fisherman's Wharf, disfrutar de la gastronomía local, 
                tomar un tranvía de la ciudad y mucho más. La ciudad es una mezcla de culturas, con una variedad de atracciones que satisfacen a viajeros de todas las edades y gustos 
                y ofrece algunos de los mejores destinos, como el famoso Golden Gate Park, el barrio de Chinatown
                y el Union Square. Por todas estas razones, San Francisco es un destino imprescindible para los viajeros.""")
    st.write(f'<iframe src="https://www.losapuntesdelviajero.com/que-ver-en-san-francisco/" width="800" height="600" style="overflow:auto"></iframe>', unsafe_allow_html=True)
    url1 = "https://www.losapuntesdelviajero.com/que-ver-en-san-francisco/"
    st.markdown("[Los apuntes del viajero](%s)" % url1)

#-------------------------------------------------------------------------------SEGUNDA PESTAÑA: INSIDE AIR BNB------------------------------------------------------------------------------------
tab_plots = tabs[1]

# mostramos la pagina de Inside Airbnb
with tab_plots:
    st.header('Inside Airbnb (Web)')
    st.write(f'<iframe src="http://insideairbnb.com/san-francisco/" width="800" height="600" style="overflow:auto"></iframe>', unsafe_allow_html=True)
    url2 = "http://insideairbnb.com/san-francisco/"
    st.markdown("[Inside Airbnb](%s)" % url2)

# mostramos el dataframe extraido de la página de Inside Airbnb
with tab_plots:
    st.header('DataSet')
    st.markdown('DataSet antes del preprocesamiento, descargado de la web Inside Airbnb')
    st.write(listingsnulls)
    url3="http://data.insideairbnb.com/united-states/ca/san-francisco/2022-12-04/data/listings.csv.gz"
    st.markdown("[Descarga el CSV](%s)" % url3)

#-------------------------------------------------------------------------------TERCERA PESTAÑA: PREPROCESAMIENTO-------------------------------------------------------------------------------------
tab_plots = tabs[2]

# importamos librerias 
with tab_plots:
    st.header('Preprocesamiento completo')
    st.subheader('*El preprocesamiento ha sido una de las partes principales de este trabajo para poder realizar el EDA*')
    st.markdown('Librerias necesarias para realizar este proyecto (Requirements)')
    code = """
    # librerias básicas
    import os
    import re
    import requests
    import pandas as pd
    import numpy as np

    # librerias machine learning para regresión lineal
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
    from sklearn import preprocessing

    # mapas interactivos
    import folium
    from folium.plugins import FastMarkerCluster

    # plotear gráficos y visualización
    import matplotlib.pyplot as plt
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.offline import plot
    import chart_studio.plotly as py

    # streamlit
    import streamlit as st
    import streamlit.components.v1 as components
    import google 
    from PIL import Image
    from pyngrok import ngrok
    from IPython.display import display
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter
    import urllib.request
    import unicodedata
    from unicodedata import name
    import plotly.io as pio

    # editor de texto
    from wordcloud import WordCloud
    """
    st.code(code, language='python')

# creamos una función para comparar los datasets que son iguales al descargar estos
with tab_plots:
    st.header('Preprocesamiento completo')
    st.subheader('*El preprocesamiento ha sido una de las partes principales de este trabajo para poder realizar el EDA*')
    st.markdown('Al descargar los datasets de la pagina de insideairbnb, comprobamos que tenemos datasets con el mismo nombre. Con lo que vamos a comparar si tienen además columnas iguales.')
    code = """
    def columns_compare(df1, df2):
        #columns_compare toma dos dataframes como entradas y compara las columnas de ambos dataframes.
        #:param: df1 (DataFrame): Primer DataFrame
        #param: df2(DataFrame): Segundo DataFrame
        #return: Lista con las columnas iguales entre los dos dataframes.
    columns_df1 = df1.columns.tolist()
    columns_df2 = df2.columns.tolist()
    equal_columns = list(set(columns_df1) & set(columns_df2))
    return equal_columns
equal_columns = columns_compare(listings, listings1)
print("Columnas iguales: ",equal_columns)
"""
    st.code(code, language='python')
    
# comparamos las columnas de los datasets con el mismo nombre, si tienen valores iguales para poder realizar algun merge
with tab_plots:
    st.markdown('Al tener columnas con los mismos nombres, comparamos una columna para ver si todos los valores son parecidos, de esta manera podremos elegir si hacer merge en alguna y juntar los dataframes.')
    code = """
    def value_colums(df1, df2):
        #value_colums toma dos dataframes como entradas y verifica si la columna de ambos dataframes tienen los mismos valores.
        #:param: df1 (DataFrame): Primer DataFrame
        #:param: df2 (DataFrame): Segundo DataFrame
        #:return: True si los valores son iguales, False si no lo son.
    values_df1 = df1['id'].values
    values_df2 = df2['id'].values
    if np.array_equal(values_df1, values_df2):
        return True
    else:
        diferents = np.where(values_df1 != values_df2)[0]
        print("Valores diferentes:", values_df2[diferents])
        return False
value_colums(listings, listings1)    
"""
    st.code(code, language='python')

# creamos una variable para conocer el % de los valores nulos
with tab_plots:
    st.markdown('Creamos una variable percent_missing y con ella ordenamos los valores nulos por porcentaje.')
    code = """
    percent_missing = listings.isnull().sum() * 100 / len(listings)
    percent_missing.sort_values(ascending = False).head()
"""
    st.code(code, language='python')

# hacemos un gráfico sobre los nulos dentro del dataset 
with tab_plots:
    nulls = listingsnulls.isnull().sum()
    fig = go.Figure(go.Bar(x=nulls.index, y=nulls))
    fig.update_layout(template="plotly_dark", xaxis_title="Columnas por indice", yaxis_title="Total Nulos")
    fig.update_layout(xaxis=dict(title_standoff=10, tickangle=45))
    st.subheader('*Distribución de valores nulos por columnas*')
    st.plotly_chart(fig)

# lista de columnas a eliminar que sobrepasan el 30% de valores nulos
with tab_plots:
    st.markdown('Eliminamos las columnas que tienen más de un 30%, añadimos un inplace = True, para fijar esta acción.')
    code = """
    columns_to_drop = ["neighbourhood_group_cleansed", "calendar_updated", "bathrooms", "license", "host_about", "host_neighbourhood", "neighborhood_overview", "neighbourhood"]
    listings.drop(columns_to_drop, axis=1, inplace=True)
"""
    st.code(code, language='python')

# creamos una función para rellenar los valores nulos por la media o la moda según el tipo de dato
with tab_plots:
    st.markdown('Convertimos los valores nulos restantes, usando la media y la moda según convenga.')
    code = """
    def fillna(df, columns):
    #fillna rellena los valores nulos en un DataFrame con la media o moda dependiendo del tipo de columna.
    #:param df: DataFrame que contiene los datos.
    #:param columns: Lista con los nombres de las columnas que contienen
    #:return: Devuelve dataframe con los valores nulos rellenados.
    for column in columns:
        if(df[column].dtype == 'float64' or df[column].dtype == 'int64'):
            number = df[column].mean()
        else:
            number = df[column].mode()[0]
        df[column]= df[column].fillna(number)
    return df
listings = fillna(listings, listings.columns)
"""
    st.code(code, language='python')

# creamos una función para eliminar los outliers
with tab_plots:
    st.markdown('Utilizamos esta función para eliminar los outliers de nuestras columnas.')
    code = """
def outliers_repair (df):
    #outliers_repair se usa para identificar y reparar valores atípicos en un DataFrame dado.
    #:param listings: DataFrame para el cual se deben reparar los valores atípicos.
    #:returns: dataFrame sin valores atípicos.
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    dataset_wo_outliers = df.copy()
    dataset_wo_outliers = pd.DataFrame(df)
    for col in df.columns:
        if df[col].dtype == 'object':
            mode = df[col].mode()[0]
            dataset_wo_outliers.loc[outliers[col], col] = mode
        else:
            mean = df[col].mean()
            dataset_wo_outliers.loc[outliers[col], col] = mean
    return dataset_wo_outliers
    listings = pd.DataFrame(outliers_repair(listings))
"""
    st.code(code, language='python')

# hacemos una función para eliminar el simbolo del $ de la columna 'price'
with tab_plots:
    st.markdown('En la columna price quitamos el simbolo $ para poder trabajar mejor con estos datos.')
    code = """
    def remove_sign(df, column):
    #remove_sign se usa para eliminar un simbolo de en este caso un número
    #:param listings: dataframe que estamos usando
    #:param price: columna que queremos usar
    #:return: dataframe con el texto eliminado de la columna
    listings['price'] = listings['price'].apply(lambda x: x.replace("$", ""))
    return listings
    listings = remove_sign(listings, 'price')
"""
    st.code(code, language='python')

# hacemos una función para eliminar las comas y convertimos el precio a float
with tab_plots:
    st.markdown('Hacemos una función para eliminar las comas y convertimos el precio a float, ya que en un futuro nos dará problemas para realizar los plots.')
    code = """
    def fix_price_column(df):
    #fix_price_column quita las comas de la columa que seleccionemos de nuestro dataframe y cambia los valores a float
    #:param df: Dataframe sobre el que vamos a usar esta función
    #:return: el valor corregido dentro de la columna asignada
    listings['price'] = listings['price'].str.replace(',','')
    listings['price'] = listings['price'].astype(float)
    return listings
listings = fix_price_column(listings)
"""
    st.code(code, language='python')

# hacemos una función para eliminar una longitud que da problemas a la hora de representar los mapas
with tab_plots:
    st.markdown('Al hacer los mapas, nos damos cuenta que una coordinada dentro de la longitud nos da problemas, por lo procedemos a borrarla.')
    code = """
    def drop_rows(df, column, value):
    #drop rows elimina el valor que queremos de una fila de una columna
    #:param df: dataframe que vamos a usar
    #:param column: columna que tendrá los valores de las filas que queremos borrar
    #:param value: valor que queremos borrar
    #:return: devuelve el dataframe 
    df = df[df[column] != value]
    return df
    listings = drop_rows(listings, "longitude",  -122.43018616682234)
    """
    st.code(code, language='python')
    

# creamos una función lambda que nos permita agrupar la distribución en los precios normales comparados con el dataframe listings que viene preparado
    with tab_plots:
        st.markdown('Creamos una función lambda que nos permita agrupar la distribución en los precios normales comparados con el dataframe listings que viene preparado.')
    code = """listings['price'] = listings['price'].apply(lambda x: int(x/1000) if x >= 10000 else (int(x/100) if x >= 1000 else x))"""
    st.code(code, language='python')

# creamos una función para eliminar dentro de la columna precio los outliers
with tab_plots:
    st.markdown('Nos damos cuenta que también tenemos que eliminar varios outliers que no se han eliminado en la columna price, para hacer bien los gráficos.')
    code = """
    def remove_outliers(df, column):
    #esta función elimina valores outliers dentro de una columna de nuestro def
    #:param df: llama al dataframe
    #:param column: columna a eliminar los outliers
    #:return: retorna el df sin outliers en nla columna seleccionada
    df_clean = df[df[column].notnull()]
    mean = df[column].mean()
    std = df[column].std()
    outliers = df[(df[column] < mean - 2*std) | (df[column] > mean + 2*std)].index # std*2 es el doble de la desviación estandar que sirve para definir un rango de valores "normales" para una variable
    df.drop(outliers, inplace=True)
    return df
    listings = remove_outliers(listings, 'price')
"""
    st.code(code, language='python')

# agrupamos los barrios en sus distritos para una mejor lectura y creamos un nueva columna dentro del dataframe
with tab_plots: 
    st.markdown('Para una mejor lectura de los datos agrupamos los barrios en distritos.')
    code = """
listings['districts'] = ''
listings.loc[listings['neighbourhood_cleansed'].isin(['Pacific Heights', 'Nob Hill', 'Presidio Heights', 'Russian Hill']), 'districts'] = 'Central/downtown'
listings.loc[listings['neighbourhood_cleansed'].isin(['Western Addition', 'Haight Ashbury', 'Glen Park', 'Downtown/Civic Center', 'Financial District', 'Marina', 'North Beach', 'Chinatown']), 'districts'] = 'Upper Market and beyond (south central)'
listings.loc[listings['neighbourhood_cleansed'].isin(['Bernal Heights', 'Bayview', 'Excelsior', 'Outer Mission', 'Inner Sunset', 'Visitacion Valley', 'Crocker Amazon', 'Ocean View', 'Parkside']), 'districts'] = 'Bernal Heights/Bayview and beyond (southeast)'
listings.loc[listings['neighbourhood_cleansed'].isin(['Mission', 'Castro/Upper Market', 'Potrero Hill', 'South of Market', 'Outer Sunset']), 'districts'] = 'Upper Market and beyond (south central)'
listings.loc[listings['neighbourhood_cleansed'].isin(['Outer Richmond', 'Inner Richmond', 'West of Twin Peaks', 'Noe Valley', 'Twin Peaks', 'Golden Gate Park', 'Presidio', 'Lakeshore', 'Diamond Heights', 'Seacliff']), 'districts'] = 'Richmond'
listings.loc[listings['neighbourhood_cleansed'].isin(['Inner Richmond', 'Inner Sunset', 'Parkside', 'Outer Sunset']), 'districts'] = 'Sunset'
"""
    st.code(code, language='python')


#-------------------------------------------------------------------------------------CUARTA PESTAÑA: EDA-----------------------------------------------------------------------
tab_plots = tabs[3]

# creamos una matriz de coorrelación entre varias variables y un mapa de correlación entre esas variables
with tab_plots:
    st.header('Análisis exploratorio de San Francisco')
    st.subheader('*Tabla de Correlación*')
    st.markdown('Creamos una matriz de correlación de estas columnas ["price", "minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count", "availability_365"]')
    corr = listings[['price','minimum_nights', 'number_of_reviews', 'reviews_per_month',
    'calculated_host_listings_count', 'availability_365']].corr()
    st.table(corr)
    st.subheader('*Matriz de Correlación*')
    fig = px.imshow(corr, color_continuous_scale=px.colors.sequential.Jet, template="plotly_dark")
    fig.update_layout(width=800)
    st.plotly_chart(fig)
    st.markdown('- Número de noches mínimas vs. Número de revisiones: -0.3594.')
    st.markdown('- Número de revisiones vs. Número de revisiones por mes: 0.3926.')
    st.markdown('- Precio vs. Número de noches mínimas: -0.1795.')
    st.markdown('- Precio vs. Número de revisiones: 0.0055.')
    st.markdown('- Precio vs. Revisiones por mes: 0.0339.')

# hacemos un plot del precio por medio de alojamientos haciendo un groupby según distrito y huéspedes
with tab_plots:
    st.subheader('*Distribución de las habitaciones en San Francisco*')
    st.markdown('Código de Llorenç Fernández')
    html = open("map.html", "r", encoding='utf-8').read()
    st.components.v1.html(html,height=600)

# plot pie de la distribución en % de los barrios en San Francisco
with tab_plots:
    st.subheader('*Distribución en % de los barrios en San Francisco*')
    value_counts = listings['neighbourhood_cleansed'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts.values, hovertext=value_counts.index)])
    fig.update_layout(template="plotly_dark")
    fig.update_layout(width=800)
    st.plotly_chart(fig)

# viendo la distribución en el gráfico de pastel, lo pasamos a un mapa
with tab_plots:
    st.subheader('*Distribución de los barrios en San Francisco*')
    map = px.scatter_mapbox(listings, lat='latitude', lon='longitude', color='neighbourhood_cleansed',
                        size_max=15, zoom=10, height=600)
    map.update_layout(mapbox_style='open-street-map')
    map.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
    map.update_layout(width=800)
    map.update_layout(template="plotly_dark")
    st.plotly_chart(map)

# plot pie de la distribución en % de los distritos en San Francisco
with tab_plots:
    st.subheader('*Distribución en % de los distritos en San Francisco*')
    value_counts = listings['districts'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts.values, hovertext=value_counts.index)])
    fig.update_layout(template="plotly_dark")
    fig.update_layout(width=800)
    st.plotly_chart(fig)

# viendo la distribución en el gráfico de pastel, lo pasamos a un mapa
with tab_plots:
    st.subheader('*Distribución de los distritos en San Francisco*')
    map = px.scatter_mapbox(listings, lat='latitude', lon='longitude', color='districts',
                        size_max=15, zoom=10, height=600)
    map.update_layout(mapbox_style='open-street-map')
    map.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
    map.update_layout(width=800)
    map.update_layout(template="plotly_dark")
    st.plotly_chart(map)

# mostramos una web donde nos da información sobre la seguridad en los barrios
with tab_plots:
    st.subheader('*Crimen por barrios en San Francisco*')
    st.write(f'<iframe src="https://www.civichub.us/ca/san-francisco/gov/police-department/crime-data" width="800" height="600" style="overflow:auto"></iframe>', unsafe_allow_html=True)
    url4 = "https://www.civichub.us/ca/san-francisco/gov/police-department/crime-data"
    st.markdown("[Civichub](%s)" % url4)

# comparamos las variables vecindario (x) y la categoria del incidente (y), con la categoria de incidente en un scatter()
with tab_plots:
    st.subheader('*Tipo de crimen por barrios*')
    fig = go.Figure(data=go.Scatter(x=security['analysis_neighborhood'], y=security['incident_category'],mode='markers'))
    fig.update_layout(template="plotly_dark")
    fig.update_layout(width=800)
    st.plotly_chart(fig)

# hacemos un plot del precio por medio de alojamientos haciendo un groupby según distrito y huéspedes
with tab_plots:
    st.subheader('*Precio promedio de alojamientos: Distrito y húespedes*')
    st.markdown('Encontramos que en la mayoría de distritos los húespedes elijen habitaciones para 2 y 4 personas.')
    listings_accommodates = listings.groupby(['districts', 'accommodates'])['price'].mean().sort_values(ascending=True).reset_index()
    fig = px.bar(listings_accommodates, x='price', y='districts', color='accommodates', orientation='h',
             labels={'price':'Precio medio', 'neighbourhood_cleansed':'Barrio', 'accommodates':'Cantidad de huéspedes'}, template="plotly_dark")
    fig.update_layout(width=800)
    st.plotly_chart(fig)
    
# creamos un grafico del precio promedio de un alojamiento para 2 personas en cada barrio
with tab_plots:
    st.subheader('*Precio promedio de un alojamiento para 2 personas en cada barrio*')
    st.markdown('La elección mayoritaria en San Francisco es de habitaciones para dos personas.')
    st.markdown('Por ello analizamos ese valor para los barrios y los distritos.')
    st.markdown('El barrio con más alojammientos para dos personas es ChinaTown seguido de Financial District')
    average = listings[listings['accommodates']==2]
    average = average.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=True)
    fig = go.Figure(data=[go.Bar(y=average.index, x=average.values, orientation='h')])
    fig.update_layout(xaxis_title='Precio medio', yaxis_title='Barrio', template="plotly_dark")
    fig.update_layout(width=800)
    st.plotly_chart(fig)
    
# creamos un grafico del precio promedio de un alojamiento para 2 personas en cada distrito
with tab_plots:
    st.subheader('*Precio promedio de un alojamiento para 2 personas en cada distrito*')
    st.markdown('El distrito con más alojamientos es Upper Market, este distrito corresponde a los barrios de ChinaTown y Financial District')
    average = listings[listings['accommodates']==2]
    average = average.groupby('districts')['price'].mean().sort_values(ascending=True)
    fig = go.Figure(data=[go.Bar(y=average.index, x=average.values, orientation='h')])
    fig.update_layout(xaxis_title='Precio medio', yaxis_title='Distrito', template="plotly_dark")
    fig.update_layout(width=800)
    st.plotly_chart(fig)
    
# hacemos un value counts para ver cuantos tipos de habitaciones hay y creamos un diagrama de barras y lo ploteamos
with tab_plots:
    st.subheader('*Tipos de habitación en San Francisco*')
    st.markdown('El tipo de habitación más común son habitaciones enteras o apartamentos')
    room_type_counts = listings.room_type.value_counts()
    fig = px.bar(room_type_counts, x=room_type_counts.index, y=room_type_counts.values,
                template='plotly_dark',
                color=room_type_counts.index,
                color_continuous_scale='Plasma',
                labels={'x':'', 'y':''})
    fig.update_layout(xaxis_title="Tipo de habitación", yaxis_title="Total")
    fig.update_layout(width=800)
    st.plotly_chart(fig)
    
# creamos un mapa que relacione la distribución de los precios y los tipos de habitación
with tab_plots:
    st.subheader('*Distribución de los precios y los tipos de habitación*')
    st.markdown('Observamos que los precios más altos los encontramos entorno a "Golden Gate Park Avenue", "Post Street", "Bay Street, y "Market Street".')
    map = px.scatter_mapbox(listings, lat="latitude", lon="longitude",
                        opacity=1.0, 
                        color =  'price', 
                        color_continuous_scale=px.colors.sequential.Jet, 
                        height = 600, zoom = 9.7,
                        text= 'room_type',
                        hover_name = 'name')
    map.update_layout(mapbox_style="open-street-map")
    map.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
    map.update_layout(template="plotly_dark")   
    map.update_layout(width=800)
    st.plotly_chart(map)

# creamos intervalos de precio para agrupar los precios que nos encontramos y que sea más fácil de lectura y hacemos la distribución de precios por barrio
with tab_plots:
    st.subheader('*Distribución de precio por barrio*')
    st.markdown('El barrio con el Airbnb más caro es Western Addition')
    st.markdown('El barrio con el Airbnb más barato es Bayview')
    listings['price_bin'] = pd.cut(listings["price"], [0,100,200,300,400,float('inf')], 
                              labels=["0-100","100-200","200-300","300-400","400-500"])
# creamos una tabla pivote entre la columna de barrios y el precio
    pivot_table = listings.pivot_table(values='id', index='neighbourhood_cleansed', columns='price_bin', aggfunc='count')
# hacemos el grafico de barras primero introduciendo la pivot table
    data = []
    for i in pivot_table.index:
        data.append(go.Bar(x=pivot_table.columns, y=pivot_table.loc[i], name=i))
# mostramos el gráfico de barras
    fig = go.Figure(data=data)
    fig.update_layout(template='plotly_dark', xaxis_title='Precio', yaxis_title='Total de casas', barmode='stack')
    fig.update_layout(width=800)
    st.plotly_chart(fig)

# creamos intervalos de precio para agrupar los precios que nos encontramos y que sea más fácil de lectura y hacemos la distribución de precios por distrito
with tab_plots:
    st.subheader('*Distribución de precio por distrito*')
    st.markdown('El distrito con los Airbnb más caros es Upper Market')
    st.markdown('El distrito con los Airbnb más baratos es Bernal Height')
    listings['price_bin'] = pd.cut(listings["price"], [0,100,200,300,400,float('inf')], 
                              labels=["0-100","100-200","200-300","300-400","400-500"])
# creamos una tabla pivote entre la columna de barrios y el precio
    pivot_table2 = listings.pivot_table(values='id', index='districts', columns='price_bin', aggfunc='count')
# hacemos el grafico de barras primero introduciendo la pivot table
    data = []
    for i in pivot_table2.index:
        data.append(go.Bar(x=pivot_table2.columns, y=pivot_table2.loc[i], name=i))
# mostramos el gráfico de barras
    fig = go.Figure(data=data)
    fig.update_layout(template='plotly_dark', xaxis_title='Precio', yaxis_title='Total de casas', barmode='stack')
    fig.update_layout(width=800)
    st.plotly_chart(fig)

# hacemos un plot sobre la cantidad de días durante los cuales un host en particular está disponible en un año
with tab_plots:
    st.subheader('*Cantidad de días durante los que un host en particular está disponible dentro del año*')
    st.markdown('La mayoría están disponible durante los 365 días del año')
    map = px.scatter_mapbox(listings, lat='latitude', lon='longitude', color='availability_365',
                        size_max=15, zoom=10, height=600, color_continuous_scale='viridis')
    map.update_layout(mapbox_style='open-street-map')
    map.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
    map.update_layout(template="plotly_dark")
    map.update_layout(width=800)
    st.plotly_chart(map)

# creamos una variable para ver que posibles alquileres ilegales encontramos
with tab_plots:
    st.subheader('*Leyes que afectan a los airbnb*')
    st.markdown("""En San Francisco, California, las rentas de Airbnb están sujetas a ciertas regulaciones. 
                Según la ordenanza de "Alquileres residenciales a corto plazo" de la ciudad, los anfitriones 
                deben registrar sus unidades con la ciudad y pagar un impuesto hotelero del 14%. 
                Los anfitriones están limitados a alquilar su residencia principal durante un máximo de 90 días por año calendario, 
                a menos que hayan obtenido un permiso de uso condicional. Las violaciones de estas regulaciones pueden resultar en multas y sanciones
                Además de las regulaciones en San Francisco, California tiene regulaciones adicionales que afectan a las rentas de Airbnb.
                Por ejemplo, la ley estatal requiere que todos los anfitriones proporcionen cierta información a los huéspedes, 
                como información de contacto de emergencia y información sobre detectores de humo y extintores""")
    st.markdown('Código de Paloma Rodríguez')
#creamos el codigo que nos de la tabla en función de las leyes que afectan
    illegal_rentings = listings.groupby(['host_id', 'host_name', 'maximum_nights']).size().reset_index(name='illegal_rooms')
    illegal_rentings = illegal_rentings.sort_values(by=['maximum_nights'], ascending=False)
    illegal_rentings = illegal_rentings[illegal_rentings['maximum_nights'] >= 90]
    st.write(illegal_rentings)
    url5 = "https://sfplanning.org/"
    st.markdown("[Sfplanning](%s)" % url5)
    url6 = "https://www.lodgify.com/guides/short-term-rental-rules-california/"
    st.markdown("[Lodgify](%s)" % url6)

# ploteamos un mapa de palabras
with tab_plots:
    st.subheader('*Mapa de palabras sobre San Francisco*')
    text = ' '.join([text for text in listings['name']])
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='black').generate(str(text))
    plt.figure( figsize=(20,10) )
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.image(wordcloud.to_image(), caption='Wordcloud')


#-------------------------------------------------------------------------------QUINTA PESTAÑA: CONCLUSIONES ------------------------------------------------------------------------------------
tab_plots = tabs[4]
with tab_plots:
    image = Image.open('C:\\Users\\rober\\Documents\\Rober\\Bootcamp\\Modulo 2\\20-Trabajo Módulo 2\\sunsetwide.png')
    st.header('Conclusiones')
    st.image(image, width=800)
    st.write('Golden Gate Park: Imagen creada con DALL-E-2')
    st.subheader('*Conclusiones EDA San Francisco*')
    st.markdown('- Airbnb es una plataforma popular de alquiler vacacional en San Francisco.')
    st.markdown('- Según los datos de Inside Airbnb, en 2021 había más de 12,000 listados activos en la ciudad.')
    st.markdown('- Pacific Heights es el barrio más caro para los alquileres de Airbnb en San Francisco, conocido por sus casas de lujo y hermosas vistas del Golden Gate Bridge y la Bahía.')
    st.markdown('- Otros barrios caros para los alquileres de Airbnb en San Francisco incluyen: Russian Hill, Presidio Heights, Presidio')
    st.markdown('- Según los datos de Inside Airbnb, en 2021 el barrio de Bayview es el más económico para los alquileres de Airbnb en San Francisco')
    st.markdown('- Alojarse en barrios como Bayview o Excelsior, pueden ofrecer una mejor experiencia para el turista, dado el ambiente local')
    st.markdown('- La ciudad tiene una tasa relativamente alta de delitos contra la propiedad, con incidentes de robo y allanamiento de morada siendo especialmente comunes.')
    st.markdown('- Las tasas de delitos violentos en San Francisco son relativamente bajas en comparación con otras ciudades importantes.')
    st.markdown('- Aunque desde 2019, San Francisco tiene una ley que hace cummplir con un límite anual de 90 días de alquiler, encontramos Airbnb que posiblemente no la respetan.')
    st.markdown('- El salario medio de un data analyst en San Francisco, según Glassdoor es de alrededor de 90,000 al año. Sin embargo, los salarios pueden variar desde alrededor de 65,000 al año hasta más de 130,000 al año')
    

#-------------------------------------------------------------------------------QUINTA PESTAÑA: REGRESIÓN LINEAL------------------------------------------------------------------------------------

# creamos una variable para codificar estas columnas al ser categoricas
columns_to_encode = ['listing_url', 'last_scraped', 'source', 'name', 'description', 'picture_url', 'host_url', 'host_name', 'host_since', 'host_location', 'host_response_time', 'host_response_rate', 'host_acceptance_rate',
                     'host_is_superhost', 'host_thumbnail_url', 'host_picture_url', 'host_verifications', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed', 'property_type', 'room_type', 'bathrooms_text',
                     'amenities', 'has_availability', 'calendar_last_scraped', 'first_review', 'last_review', 'instant_bookable', 'districts']

# creamos un bucle for que itera por todas las columnas y las codifica
for column in columns_to_encode:
    encode = preprocessing.LabelEncoder()
    encode.fit(listings[column])
    listings[column] = encode.transform(listings[column])
listings.sort_values(by='price',ascending=True,inplace=True)

# ajustamos el modelo de regresión lineal y lo dividimos en datos de conjunto de entrenamiento y un conjunto de prueba haciendo un train_test_split
l_reg = LinearRegression()
X = listings[['id','neighbourhood_cleansed','latitude','longitude','room_type','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']]
y = listings['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
l_reg.fit(X_train,y_train)

# calculamos las diferentes medidas de rendimiento del modelo tales como Mean Squared Error, R2 Score, Mean Absolute Error, Mean Squareroot Error.
predicts = l_reg.predict(X_test)
print("Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicts)))
print("R2 Score: ", r2_score(y_test,predicts)*100)
print("Mean Absolute Error: ", mean_absolute_error(y_test,predicts))
print("Mean Squareroot Error: ", mean_squared_error(y_test,predicts))

# creamos un dataframe para comparar los valores actuales y los valores predichos
lr_pred_df = pd.DataFrame({
    #creamos un marco de datos pandas a partir de dos arrays: uno que contiene los valores reales de prueba y otro que contiene los valores predichos para las pruebas. Luego, el marco de datos se usa para mostrar las primeras 20 filas.
'actual_values': np.array(y_test).flatten(),
'predicted_values': predicts.flatten()}).head(20)

tab_plots = tabs[5]

# creamos una función para comparar los datasets que son iguales al descargar estos
with tab_plots:
    st.header('Regresión lineal: Precios')
    st.subheader('*Regresión lineal basada en Dwi Gustin Nurdialit*')
    url6 = "https://medium.com/analytics-vidhya/python-exploratory-data-analysis-eda-on-nyc-airbnb-cbeabd622e30"
    st.markdown("[En línea](%s)" % url6)
    st.markdown('Vamos a utilizar este modelo matemático para explicar la relación entre varias variables independientes y varias variables dependientes')
    st.markdown('*Con esto intentaremos predecir el valor del precio en función de las otras variables independientes*')

# buscamos las columnas tipo objeto para codificarlas
with tab_plots:
    st.markdown('Buscamos las columnas tipo objeto para codificarlas')
    code = """
searching_dtype = listings.dtypes == object
list(listings.loc[:,searching_dtype])
"""
    st.code(code)

# codificamos las variables categoricas
with tab_plots:
    st.markdown('Codificamos las variables categoricas con un bucle for que itera sobre ellas')
    code = """
    # creamos una variable para codificar estas columnas al ser categoricas
    columns_to_encode = ['listing_url', 'last_scraped', 'source', 'name', 'description', 'picture_url', 'host_url', 'host_name', 'host_since', 'host_location', 'host_response_time', 'host_response_rate', 'host_acceptance_rate',
                     'host_is_superhost', 'host_thumbnail_url', 'host_picture_url', 'host_verifications', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed', 'property_type', 'room_type', 'bathrooms_text',
                     'amenities', 'has_availability', 'calendar_last_scraped', 'first_review', 'last_review', 'instant_bookable', 'districts']
    # creamos un bucle for que itera por todas las columnas y las codifica
    for column in columns_to_encode:
    encode = preprocessing.LabelEncoder()
    encode.fit(listings[column])
    listings[column] = encode.transform(listings[column])
    listings.sort_values(by='price',ascending=True,inplace=True)    
"""
    st.code(code, language='python')

# ajustamos el modelo de regresión lineal y lo dividimos en datos de conjunto de entrenamiento y un conjunto de prueba haciendo un train_test_split
with tab_plots:
    st.markdown('Ajustamos el modelo de regresión lineal y lo dividimos en datos de conjunto de entrenamiento y un conjunto de prueba haciendo un train_test_split')
    code = """
    l_reg = LinearRegression()
    X = listings[['id','neighbourhood_cleansed','latitude','longitude','room_type','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']]
    y = listings['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    l_reg.fit(X_train,y_train)
"""
    st.code(code, language='python')

# calculamos las diferentes medidas de rendimiento del modelo tales como Mean Squared Error, R2 Score, Mean Absolute Error, Mean Squareroot Error.
with tab_plots:
    st.markdown('Calculamos las diferentes medidas de rendimiento del modelo tales como Mean Squared Error, R2 Score, Mean Absolute Error, Mean Squareroot Error.')
    code = """
    predicts = l_reg.predict(X_test)
    print("Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicts)))
    print("R2 Score: ", r2_score(y_test,predicts)*100)
    print("Mean Absolute Error: ", mean_absolute_error(y_test,predicts))
    print("Mean Squareroot Error: ", mean_squared_error(y_test,predicts))
"""
    st.code(code, language='python')
    st.markdown("""**Mean Squared Error (MSE): Es una medida de la diferencia entre los valores predichos y los valores actuales. Un valor bajo de MSE indica que el modelo está haciendo predicciones precisas.**
- El MSE es 95.91673833503985 es un valor muy alto, por lo que indica mala predicción de los datos, ya que hay una gran diferencia entre los precios.

**R2 Score: Es una medida de cuán bien el modelo es capaz de explicar la variación en los datos. Un valor cercano a 1 indica un buen rendimiento del modelo, mientras que un valor cercano a 0 indica un rendimiento pobre.**
- El R2 Score explica solo el 2.4139739201838206% de la variación en los datos. Esto sugiere que el modelo no tiene un buen ajuste para los datos y no es capaz de predecir con precisión los valores objetivos a partir de los valores de entrada.

**Mean Absolute Error (MAE): Es una medida de la diferencia entre los valores predecidos y los valores reales. Es el promedio de la magnitud de las diferencias entre los valores predecidos y los valores reales..**
- El MAE es 75.38753534057028 indica que el modelo tiene un error medio de 75.38753534057028 unidades en sus predicciones. Un modelo con un MAE muy alto no es necesariamente un mal modelo, pero indica que las predicciones del modelo son menos precisas.

**Mean Squareroot Error (MSE): Es una medida de la diferencia entre los valores predichos y los valores actuales. Un valor bajo de MSE indica que el modelo está haciendo predicciones precisas.**
- Un valor de MSRE de 9200.020692832502 indica que el modelo tiene un error medio de la raíz cuadrada de 9200.020692832502 unidades en sus predicciones. Es una medida de la precisión de las predicciones. A menor valor MSRE, mejor es la precisión del modelo.""")

# creamos un dataframe para comparar los valores actuales y los valores predichos
with tab_plots:
    st.markdown('Creamos un dataframe para comparar los valores actuales y los valores predichos y dos arrays con los valores reales y predichos')
    code = """
    # creamos un dataframe para comparar los valores actuales y los valores predichos
    lr_pred_df = pd.DataFrame({
    #creamos un marco de datos pandas a partir de dos arrays: uno que contiene los valores reales de prueba y otro que contiene los valores predichos para las pruebas. Luego, el marco de datos se usa para mostrar las primeras 20 filas.
        'actual_values': np.array(y_test).flatten(),
        'predicted_values': predicts.flatten()}).head(20)"""
    st.code(code, language='python')

with tab_plots:
# calculamos la pendiente y la intersección con el eje y de la recta de regresión
    slope, y_intercept = np.polyfit(lr_pred_df.actual_values, lr_pred_df.predicted_values, 1)
    # calculamos los valores de x e y para la regresión lineal
    reg_x = lr_pred_df.actual_values
    reg_y = slope * reg_x + y_intercept

    # ploteamos un scatter con puntos de dispersion y la linea de regresión, viendo los valores de precios que nos predice
    fig = go.Figure(data=[
        go.Scatter(x=lr_pred_df.actual_values, y=lr_pred_df.predicted_values, mode='markers', name = 'Puntos de Dispersión', marker=dict(color='red')),
        go.Scatter(x=reg_x, y=reg_y, mode='lines', name='Recta de Regresión', marker=dict(color='green'))
    ])

    fig.update_layout(
        xaxis_title='Precios Actuales',
        yaxis_title='Precios Predicción',
        template='plotly_dark',
    )
    st.subheader('*Regresión Lineal: Precios Actuales  Vs Precios Predicción*')
    st.plotly_chart(fig)
    st.markdown('**CONCLUSIÓN: Es posible que sea necesario ajustar el modelo o utilizar una técnica diferente para mejorar la precisión de las predicciones**')
    st.markdown('**Además el tipo de dato usado puede que no sea el mejor para este tipo de modelo o incluso para hacer una regresión lineal**')


#----------------------------------------------------------------------------------END----------------------------------------------------------------------------------------------------------------


