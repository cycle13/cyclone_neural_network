import scrapy
import os 

class CycloneSpider(scrapy.Spider):
    name = 'cyclone'
    filename = 'cyclone.html'

    start_urls = [
            'https://www.nrlmry.navy.mil/tcdat/'
        ]

    # Method to extract tropical cyclone for each year available
    def parse(self, response):
        open(self.filename, 'w').close()
        raw = response.css('a::attr(href)').extract()
        years = []
        for item in raw:
            if item[:2] == 'tc' and item[2:4].isnumeric() and item[4] == '/':
                years.append(item)

        for item in years:
            year = response.urljoin(item)
            yield scrapy.Request(year, callback = self.parse_basin)
        self.log(years)
    
    # Method to extract tropical cyclone for each basin
    def parse_basin(self, response):
        raw = response.css('a::attr(href)').extract()
        basins = []
        for item in raw:
            if item[:-1].isupper() and item[:-1].isalpha() and item[-1] == '/':
                basins.append(item)
        if len(basins) > 0:
            for item in basins:
                basin = response.urljoin(item)
                yield scrapy.Request(basin, callback = self.parse_cyclone)

    # Method to extract tropical cyclone name for each cyclone
    def parse_cyclone(self, response):
        raw = response.css('a::attr(href)').extract()
        cyclones = []
        for item in raw:
            if item[:2].isnumeric() and item[3] == '.' and item[4:-1].isupper() and item[4:-1].isalpha() and item[-1] == '/':
                cyclones.append(item)
        if len(cyclones) > 0:
            for item in cyclones:
                cyclone = response.urljoin(item+'ir/geo/1km_bw/')
                yield scrapy.Request(cyclone, callback = self.parse_image, errback = None)

    # Method to extract tropical cyclone images for each cyclone
    def parse_image(self, response):
        raw = response.css('a::attr(href)').extract()
        images = []
        for item in raw:
            if item[-4:] == '.jpg':
                images.append(item)

        with open(self.filename, 'a') as f:
            for item in images:
                full_url = response.urljoin(item)
                f.writelines('%s\n' % full_url)
        self.log('Saved file %s' % self.filename)