from selenium import webdriver

options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 无头模式（可选）
driver = webdriver.Chrome(options=options)

driver.get("https://www.baidu.com")
print("页面标题:", driver.title)
driver.quit()