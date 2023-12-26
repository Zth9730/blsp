from petrel_client.client import Client




if __name__ == "__main__":
	url = 'youtubeBucket/videos/--I52m-lHzc.mp4'
	client = MyClient()
	data = client.get(url)
	with io.BytesIO(data) as fobj:
		print(fobj)