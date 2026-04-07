# Import Matplotlib, pandas, and plotly
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

df1 = pd.read_csv("data/brasil-real-estate-1.csv")
df1.head()

df1.dropna(inplace=True)

df1[["lat", "lon"]] = df1["lat-lon"].str.split(",", expand=True).astype(float)

df1["state"] = df1["place_with_parent_names"].str.split("|", expand=True)[2]

df1["price_usd"] = (
    df1["price_usd"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "")
    .astype(float)
)

df1.drop(columns=["lat-lon", "place_with_parent_names"], inplace=True)

df2 = pd.read_csv("data/brasil-real-estate-2.csv")

df2["price_usd"] = (df2["price_brl"] / 3.19).round(2)
df2.drop(columns="price_brl", inplace=True)

df2.dropna(inplace=True)




df = pd.concat([df1, df2], ignore_index=True)

fig = px.scatter_map(
    df,
    lat="lat",
    lon="lon",
    center={"lat": -14.2, "lon": -51.9},
    width=600,
    height=600,
    hover_data=["price_usd"],
)

fig.update_layout(map_style="open-street-map")

# Summary stats and plotting
mean_price_by_region = df.groupby("region")["price_usd"].mean()

fig, ax = plt.subplots()
mean_price_by_region.plot(kind="bar",
                          xlabel="Region",
                          ylabel="Mean Price [USD]",
                          title="Mean Home Price by Region", ax=ax)

df_south = df[df["region"] == "South"]
df_south_rgs = df[df["state"] == "Rio Grande do Sul"]

fig, ax = plt.subplots()
ax.scatter(x=df_south_rgs["area_m2"], y=df_south_rgs["price_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")
plt.title("Rio Grande do Sul: Price vs. Area")

plt.show()
