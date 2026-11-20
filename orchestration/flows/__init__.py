"""Prefect flows untuk pipeline IHSG.

Flow sengaja TIPIS: ia hanya memanggil fungsi di ``ihsg.pipelines``. Semua
logika ada di paket, sehingga:

* flow bisa dijalankan tanpa Prefect (``python -m ihsg.cli run``),
* pindah ke Azure Container Apps Job / cron tidak memerlukan perubahan kode,
* retry Prefect tidak menggandakan tulisan karena setiap tahap idempoten.
"""
