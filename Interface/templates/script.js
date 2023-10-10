
let mapOptions = {
  center: [33.7243396617476, -40.64062714576721],
  zoom: 2,
  minZoom:2,
worldCopyJump : false,
   maxBounds : [[-90, -180],[90, 180]]
};

let map = new L.map('map', mapOptions);

let layer = new L.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    }).addTo(map);
map.addLayer(layer);

async function isMarkerOnSea(lat, lng) {
  const response = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json&accept-language=en`);
  const data = await response.json();
  // Check if the first result returned by the API is a sea or ocean
  console.log(data.error=="Unable to geocode");
    console.log(data);

  return data.error=="Unable to geocode" ;
}

let marker = null;
map.on('click', async (event) => {
  if (await isMarkerOnSea(event.latlng.lat, event.latlng.lng)) {
    if (marker !== null) {
      map.removeLayer(marker);
    }
    marker = L.marker([event.latlng.lat, event.latlng.lng]).addTo(map);
    document.getElementById('latitude').value = event.latlng.lat;
     document.getElementById('longitude').value = event.latlng.lng;
  } else {
      alert('Le marqueur doit être placer sur les océans');
  }
});