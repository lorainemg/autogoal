FROM autogoal/autogoal

COPY "mtl.requirements.txt" "."
RUN pip install -r mtl.requirements.txt --index-url http://nexus.prod.uci.cu/repository/pypi-proxy/simple/ --trusted-host nexus.prod.uci.cu