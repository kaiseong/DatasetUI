DatasetUI HTTPS certificate setup (Ubuntu)
==========================================

1. Keep this file, datasetui-root-ca.crt, and install-datasetui-ca.sh together.
2. Run: bash install-datasetui-ca.sh
3. Enter this Ubuntu computer's sudo password when prompted.
4. Fully close and reopen the browser.
5. Open: https://192.168.0.3

This is a one-time HTTPS trust setup. It is unrelated to the SSH password used
when copying a dataset to a user's computer.
