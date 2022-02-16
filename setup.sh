apt update
apt-get install -y gfortran libblas-dev liblapack-dev graphviz
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml