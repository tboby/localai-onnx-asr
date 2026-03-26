.DEFAULT_GOAL := install

.PHONY: install
install:
	bash install.sh

.PHONY: protogen-clean
protogen-clean:
	$(RM) backend_pb2.py backend_pb2_grpc.py

.PHONY: clean
clean: protogen-clean
	rm -rf __pycache__ venv
