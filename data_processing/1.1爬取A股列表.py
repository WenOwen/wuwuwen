from data_processing.samequant_functions_new import OptimizedDownloadStocksList

# 从交易所官网下载最新的股票列表（含退市和终止上市）
D_stock_lst_1 = OptimizedDownloadStocksList()
D_stock_lst_1.main()
