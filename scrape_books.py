import requests
import pandas as pd
from bs4 import BeautifulSoup

base_url = "https://books.toscrape.com/catalogue/page-{}.html"
main_site = "https://books.toscrape.com/catalogue/"

books_data = []

# get first five page
for page in range(1, 6):
    #i would do this the number of pages that exist
    print(f"Extracting for page {page}") 
    response = requests.get(base_url.format(page))
    soup = BeautifulSoup(response.text, "html.parser")
    # extract books from the page
    books = soup.find_all("article", class_="product_pod")
    print(len(books))

for book in books:
    title = book.h3.a["title"] #book name
    link = main_site + book.h3.a["href"].replace("../../../", "")
    product_page = requests.get(link)
    product_soup = BeautifulSoup(product_page.text, "html.parser")
    

    # book price
    price = product_page.find("p", class_="price_color").text.strip()

    # Stock status
    stock = product_page.find("p", class_ = "instock availability").text.strip()

    #product rating
    rating_tag = product_page.find("p", class_="star-rating")
    rating = rating_tag["class"][1].capitalize()

    #description
    description = product_soup.find("div", id="product_description")
    if description:
        description = description.find_next_sibling("p").text.strip()
        print(f"{description}\n")
    else:
        description = "No description"
    

    #product information
    product_information_tag = product_soup.find("table", class_="table table-stripped")
    product_info = {}
    for row in product_information_tag:
        key = row.th.text.strip()
        value = row.td.text.strip()
        product_info[key] = value
    #category
    category_info = product_soup.find("ul", class_="breadcrumb").find_all("a")[2].text.strip()

    books_data.append(
        {
            "Book name": title,
            "Book price":price,
            "Book Stock": stock,
            "Rating": rating,
            "Description": description,
            "Product info": product_info,
            "Category": category_info
        }
    )

df = pd.DataFrame(books_data)
df.to_csv("books_data.csv", index= False)
print("✅ Scrapping complete")



    
